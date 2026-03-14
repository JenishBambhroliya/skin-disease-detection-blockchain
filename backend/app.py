import os
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, session, flash, send_from_directory, jsonify
from functools import wraps

# Setup Path
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.config import Config
from backend.logger import app_logger
from database.db import init_db, db
from database.models import User, ImageRecord, Prediction, BlockchainRecord, AccessLog
from encryption.aes_cipher import AESCipher
from utils.upload_util import validate_and_save_upload
from utils.hash_util import generate_file_hash
from storage.ipfs_client import IPFSClient
from blockchain.blockchain_service import BlockchainService
from model.predict import SkinDiseasePredictor

app = Flask(__name__, template_folder='../frontend/templates', static_folder='../frontend/static')
app.config.from_object(Config)

# Initialize singletons
init_db(app)
cipher = AESCipher()
ipfs_client = IPFSClient()
blockchain_service = BlockchainService()
predictor = SkinDiseasePredictor()

# --- Auth Decorators ---
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash("Please log in to access this page.", "warning")
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session or session.get('role') != 'admin':
            flash("Admin access required.", "danger")
            return redirect(url_for('dashboard'))
        return f(*args, **kwargs)
    return decorated_function

def log_access(user_id, action):
    try:
        log = AccessLog(user_id=user_id, action=action, ip_address=request.remote_addr)
        db.session.add(log)
        db.session.commit()
    except Exception as e:
        app_logger.error(f"Failed to log access for user {user_id}: {e}")

# --- Routes ---
@app.route('/')
def index():
    if 'user_id' in session:
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        
        # Check if user exists
        if User.query.filter_by(username=username).first() or User.query.filter_by(email=email).first():
            flash("Username or Email already exists.", "danger")
            return redirect(url_for('register'))
            
        new_user = User(username=username, email=email)
        # First registered user becomes admin automatically for testing purposes
        if User.query.count() == 0:
            new_user.role = 'admin'
            
        new_user.set_password(password)
        
        db.session.add(new_user)
        db.session.commit()
        
        log_access(new_user.user_id, "Registered account")
        flash("Registration successful. Please log in.", "success")
        return redirect(url_for('login'))
        
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        user = User.query.filter_by(username=username).first()
        
        if user and user.check_password(password):
            session['user_id'] = user.user_id
            session['username'] = user.username
            session['role'] = user.role
            log_access(user.user_id, "Logged in")
            
            if user.role == 'admin':
                return redirect(url_for('admin_dashboard'))
            return redirect(url_for('dashboard'))
            
        flash("Invalid username or password.", "danger")
    return render_template('login.html')

@app.route('/logout')
def logout():
    log_access(session.get('user_id'), "Logged out")
    session.clear()
    flash("You have been logged out.", "success")
    return redirect(url_for('login'))

@app.route('/dashboard')
@login_required
def dashboard():
    return render_template('dashboard.html')

@app.route('/upload-image', methods=['GET', 'POST'])
@login_required
def upload_image():
    if request.method == 'POST':
        if 'image' not in request.files:
            flash('No file part', "danger")
            return redirect(request.url)
            
        file = request.files['image']
        if file.filename == '':
            flash('No selected file', "danger")
            return redirect(request.url)
            
        try:
            # 1. Save and validate raw image
            user_id = session['user_id']
            saved_path = validate_and_save_upload(file, user_id)
            
            # 2. Hash raw image
            img_hash = generate_file_hash(saved_path)
            
            # 3. Create DB record placeholder
            img_record = ImageRecord(
                user_id=user_id,
                original_filename=file.filename,
                encrypted_path="pending", # Updated below
                sha256_hash=img_hash
            )
            db.session.add(img_record)
            db.session.flush() # Get ID
            
            # 4. Predict
            pred_class, conf, desc = predictor.predict(saved_path)
            pred_record = Prediction(image_id=img_record.image_id, predicted_class=pred_class, confidence_score=conf, description=desc)
            db.session.add(pred_record)
            
            # 5. Encrypt Image
            enc_path = saved_path + ".enc"
            cipher.encrypt_file(saved_path, enc_path)
            img_record.encrypted_path = enc_path
            
            # 6. Delete raw image securely
            if os.path.exists(saved_path):
                os.remove(saved_path)
                
            # 7. Upload to IPFS
            cid = ipfs_client.upload_file(enc_path)
            img_record.ipfs_cid = cid
            
            # 8. Blockchain Record
            tx_hash = blockchain_service.add_record(user_id, img_record.image_id, img_hash, cid)
            
            if tx_hash:
                 bc_record = BlockchainRecord(image_id=img_record.image_id, transaction_hash=tx_hash, status="success")
                 db.session.add(bc_record)
                 flash("Blockchain verification successful.", "success")
            else:
                 flash("Blockchain transaction failed. Record stored locally.", "warning")
                 
            db.session.commit()
            log_access(user_id, f"Uploaded image {img_record.image_id}")
            
            return redirect(url_for('result', image_id=img_record.image_id))
            
        except Exception as e:
            app_logger.error(f"Error during upload: {str(e)}")
            flash(f"Error processing upload: {str(e)}", "danger")
            return redirect(request.url)
            
    return render_template('upload.html')

@app.route('/result/<int:image_id>')
@login_required
def result(image_id):
    record = ImageRecord.query.filter_by(image_id=image_id, user_id=session['user_id']).first_or_404()
    prediction = Prediction.query.filter_by(image_id=image_id).first()
    blockchain = BlockchainRecord.query.filter_by(image_id=image_id).first()
    return render_template('result.html', record=record, prediction=prediction, blockchain=blockchain)

@app.route('/history')
@login_required
def history():
    records = ImageRecord.query.filter_by(user_id=session['user_id']).order_by(ImageRecord.uploaded_at.desc()).all()
    # Eager load predictions to display
    for r in records:
        r.pred = Prediction.query.filter_by(image_id=r.image_id).first()
    return render_template('history.html', records=records)

@app.route('/verify-image/<int:image_id>')
@login_required
def verify_image(image_id):
    """
    Shows verification status. To do the actual comparison,
    we fetch from blockchain, decrypt the local file (or download from IPFS), 
    re-hash the decrypted file, and compare.
    """
    record = ImageRecord.query.filter_by(image_id=image_id, user_id=session['user_id']).first_or_404()
    
    # Decrypt to temp space temporarily to verify hash
    temp_path = record.encrypted_path + ".decrypted"
    try:
        cipher.decrypt_file(record.encrypted_path, temp_path)
        recomputed_hash = generate_file_hash(temp_path)
        os.remove(temp_path)
    except Exception as e:
        app_logger.error(f"Decryption failed during verification: {e}")
        recomputed_hash = "Decryption Failed"

    # Get Blockchain data
    bc_data = blockchain_service.get_record(image_id)
    
    tampered = False
    bc_hash = None
    if bc_data:
        bc_hash = bc_data['sha256Hash']
        if bc_hash != recomputed_hash:
            tampered = True
    else:
        app_logger.warning("No blockchain data found on node")
    
    return render_template('verify.html', 
                            record=record, 
                            recomputed_hash=recomputed_hash, 
                            bc_hash=bc_hash, 
                            tampered=tampered,
                            bc_data=bc_data)

@app.route('/image-preview/<int:image_id>')
@login_required
def image_preview(image_id):
    """Decrypts image temporarily, applies enhancement, and serves the decrypted raw content"""
    record = ImageRecord.query.filter_by(image_id=image_id, user_id=session['user_id']).first_or_404()
    temp_path = record.encrypted_path + ".temp.png"
    
    try:
        cipher.decrypt_file(record.encrypted_path, temp_path)
        
        # Apply enhancement
        import cv2
        from model.preprocess import enhance_image
        img = cv2.imread(temp_path)
        if img is not None:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            enhanced = enhance_image(img_rgb)
            enhanced_bgr = cv2.cvtColor(enhanced, cv2.COLOR_RGB2BGR)
            cv2.imwrite(temp_path, enhanced_bgr)

        directory = os.path.abspath(os.path.dirname(temp_path))
        filename = os.path.basename(temp_path)
        
        # We need a robust cleanup here normally. For now, sending the file securely
        return send_from_directory(directory, filename)
    except Exception as e:
        app_logger.error(f"Error serving preview: {e}")
        return "Image not available", 404

# --- Admin Routes ---
@app.route('/admin-dashboard')
@admin_required
def admin_dashboard():
    user_count = User.query.count()
    image_count = ImageRecord.query.count()
    users = User.query.all()
    return render_template('admin.html', u_count=user_count, i_count=image_count, users=users)

@app.route('/admin/promote/<int:user_id>', methods=['POST'])
@admin_required
def promote_user(user_id):
    user = User.query.get_or_404(user_id)
    if user.role != 'admin':
        user.role = 'admin'
        db.session.commit()
        log_access(session.get('user_id'), f"Promoted user {user_id} to admin")
        flash(f"User {user.username} has been promoted to Admin.", "success")
    return redirect(url_for('admin_dashboard'))

@app.route('/admin/demote/<int:user_id>', methods=['POST'])
@admin_required
def demote_user(user_id):
    user = User.query.get_or_404(user_id)
    # Prevent admin from demoting themselves
    if user.user_id == session.get('user_id'):
        flash("You cannot revoke your own admin access.", "danger")
        return redirect(url_for('admin_dashboard'))
    if user.role == 'admin':
        user.role = 'user'
        db.session.commit()
        log_access(session.get('user_id'), f"Revoked admin access for user {user_id}")
        flash(f"Admin access revoked for {user.username}.", "success")
    return redirect(url_for('admin_dashboard'))

@app.route('/admin/users')
@admin_required
def admin_users():
    users = User.query.all()
    # Simple JSON representation for admin
    return jsonify([{
        'id': u.user_id,
        'username': u.username,
        'email': u.email,
        'role': u.role
    } for u in users])

@app.route('/admin/images')
@admin_required
def admin_images():
    images = ImageRecord.query.all()
    # Simple JSON representation for admin
    return jsonify([{
        'id': img.image_id,
        'user_id': img.user_id,
        'hash': img.sha256_hash,
        'ipfs_cid': img.ipfs_cid
    } for img in images])

@app.route('/ipfs/<cid>')
@login_required
def ipfs_gateway(cid):
    """Serves files stored locally by their IPFS CID.
    Decrypts the encrypted file and serves it as an image."""
    from flask import Response, after_this_request
    
    ipfs_local_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'storage', 'ipfs_local')
    file_path = os.path.join(ipfs_local_dir, cid)
    
    if os.path.exists(file_path):
        temp_path = file_path + ".temp_view"
        try:
            cipher.decrypt_file(file_path, temp_path)
            
            with open(temp_path, 'rb') as f:
                data = f.read()
            
            # Clean up temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)
            
            # Detect image type from magic bytes
            mime = 'application/octet-stream'
            if data[:3] == b'\xff\xd8\xff':
                mime = 'image/jpeg'
            elif data[:8] == b'\x89PNG\r\n\x1a\n':
                mime = 'image/png'
            elif data[:4] == b'GIF8':
                mime = 'image/gif'
            
            return Response(data, mimetype=mime, headers={
                'Content-Disposition': f'inline; filename={cid}.jpg'
            })
        except Exception as e:
            app_logger.error(f"Failed to serve IPFS file {cid}: {e}")
            if os.path.exists(temp_path):
                os.remove(temp_path)
            return f"Error decrypting file: {str(e)}", 500
    else:
        return f"CID {cid} not found in local IPFS store", 404


if __name__ == '__main__':
    app.run(debug=True, port=5000)
