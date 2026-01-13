"""
Serveur Flask pour le modèle BRIA-RMBG (Background Removal)
À déployer sur un serveur cloud (Railway, Render, Google Cloud Run, etc.)

Installation:
    pip install flask torch torchvision pillow transformers

Pour BRIA-RMBG spécifiquement:
    pip install transparent-background
    ou
    pip install rembg  # alternative si transparent-background pose problème
"""

from flask import Flask, request, send_file, jsonify
from PIL import Image
import io
import os
import torch
from transformers import pipeline
import logging

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Initialisation du modèle BRIA-RMBG
# Option 1: Utiliser le pipeline Hugging Face officiel de BRIA
rmbg_pipe = None

def load_model():
    """Charge le modèle BRIA-RMBG"""
    global rmbg_pipe
    try:
        # BRIA RMBG v1.4 - Le modèle officiel
        rmbg_pipe = pipeline(
            "image-segmentation",
            model="briaai/RMBG-1.4",
            trust_remote_code=True,
            device=0 if torch.cuda.is_available() else -1
        )
        logger.info("Modèle BRIA-RMBG chargé avec succès")
    except Exception as e:
        logger.error(f"Erreur lors du chargement du modèle: {e}")
        raise

def remove_background_bria(image_bytes):
    """
    Supprime le fond d'une image en utilisant BRIA-RMBG
    
    Args:
        image_bytes: bytes de l'image d'entrée
    
    Returns:
        bytes de l'image PNG avec fond transparent
    """
    # Charger l'image
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    
    # Appliquer le modèle BRIA-RMBG
    result = rmbg_pipe(image)
    
    # Le résultat est une image avec masque
    # Extraire l'image avec fond transparent
    if isinstance(result, list):
        # Si plusieurs résultats, prendre le premier
        output_image = result[0]['mask'] if 'mask' in result[0] else result[0]
    else:
        output_image = result
    
    # Si le résultat est un masque, l'appliquer à l'image originale
    if isinstance(output_image, Image.Image):
        # Créer une image RGBA
        rgba_image = image.convert("RGBA")
        
        # Convertir le masque en mode L si nécessaire
        if output_image.mode != 'L':
            mask = output_image.convert('L')
        else:
            mask = output_image
        
        # Redimensionner le masque si nécessaire
        if mask.size != rgba_image.size:
            mask = mask.resize(rgba_image.size, Image.LANCZOS)
        
        # Appliquer le masque comme canal alpha
        rgba_image.putalpha(mask)
        output_image = rgba_image
    
    # Sauvegarder en PNG
    output_buffer = io.BytesIO()
    output_image.save(output_buffer, format='PNG', optimize=True)
    output_buffer.seek(0)
    
    return output_buffer

@app.route('/health', methods=['GET'])
def health_check():
    """Endpoint de vérification de santé"""
    return jsonify({
        "status": "healthy",
        "model": "BRIA-RMBG-1.4",
        "gpu_available": torch.cuda.is_available()
    })

@app.route('/remove-background', methods=['POST'])
def remove_background():
    """
    Endpoint principal pour la suppression de fond
    
    Attend une image en multipart/form-data avec la clé 'image'
    Retourne l'image PNG avec fond transparent
    """
    try:
        # Vérifier que le modèle est chargé
        if rmbg_pipe is None:
            load_model()
        
        # Vérifier la présence du fichier
        if 'image' not in request.files:
            return jsonify({"error": "Aucune image fournie. Utilisez le champ 'image'"}), 400
        
        file = request.files['image']
        
        if file.filename == '':
            return jsonify({"error": "Nom de fichier vide"}), 400
        
        # Lire les bytes de l'image
        image_bytes = file.read()
        
        # Vérifier que c'est bien une image
        try:
            Image.open(io.BytesIO(image_bytes))
        except Exception:
            return jsonify({"error": "Fichier non reconnu comme image valide"}), 400
        
        logger.info(f"Traitement de l'image: {file.filename}")
        
        # Supprimer le fond
        output_buffer = remove_background_bria(image_bytes)
        
        # Générer le nom de fichier de sortie
        original_name = os.path.splitext(file.filename)[0]
        output_filename = f"{original_name}_nobg.png"
        
        logger.info(f"Image traitée avec succès: {output_filename}")
        
        return send_file(
            output_buffer,
            mimetype='image/png',
            as_attachment=True,
            download_name=output_filename
        )
    
    except Exception as e:
        logger.error(f"Erreur lors du traitement: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/remove-background-base64', methods=['POST'])
def remove_background_base64():
    """
    Endpoint alternatif acceptant une image en base64
    
    Body JSON: {"image": "base64_string", "filename": "nom.jpg"}
    Retourne l'image en base64
    """
    import base64
    
    try:
        if rmbg_pipe is None:
            load_model()
        
        data = request.get_json()
        
        if not data or 'image' not in data:
            return jsonify({"error": "Champ 'image' requis (base64)"}), 400
        
        # Décoder le base64
        image_bytes = base64.b64decode(data['image'])
        
        # Traiter l'image
        output_buffer = remove_background_bria(image_bytes)
        
        # Encoder le résultat en base64
        output_base64 = base64.b64encode(output_buffer.read()).decode('utf-8')
        
        return jsonify({
            "success": True,
            "image": output_base64,
            "format": "png"
        })
    
    except Exception as e:
        logger.error(f"Erreur: {str(e)}")
        return jsonify({"error": str(e)}), 500

# Alternative avec transparent-background (si le pipeline HF pose problème)
def remove_background_transparent_bg(image_bytes):
    """
    Alternative utilisant la bibliothèque transparent-background
    """
    from transparent_background import Remover
    
    remover = Remover()
    
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    output = remover.process(image)
    
    output_buffer = io.BytesIO()
    output.save(output_buffer, format='PNG')
    output_buffer.seek(0)
    
    return output_buffer

if __name__ == '__main__':
    # Charger le modèle au démarrage
    load_model()
    
    # Lancer le serveur
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=False)
