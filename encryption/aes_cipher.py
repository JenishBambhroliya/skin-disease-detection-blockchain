import os
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives import padding
from cryptography.hazmat.backends import default_backend
from backend.config import Config

class AESCipher:
    def __init__(self):
        # AES-256 key must be 32 bytes
        self.key = Config.AES_SECRET_KEY
        if len(self.key) != 32:
            raise ValueError("AES_SECRET_KEY must be exactly 32 bytes for AES-256")
        self.backend = default_backend()

    def encrypt_file(self, input_path: str, output_path: str):
        """Encrypts a file and saves exactly to output_path."""
        iv = os.urandom(16)
        cipher = Cipher(algorithms.AES(self.key), modes.CBC(iv), backend=self.backend)
        encryptor = cipher.encryptor()

        with open(input_path, 'rb') as f_in:
            data = f_in.read()

        padder = padding.PKCS7(128).padder()
        padded_data = padder.update(data) + padder.finalize()
        encrypted_data = encryptor.update(padded_data) + encryptor.finalize()

        with open(output_path, 'wb') as f_out:
            f_out.write(iv + encrypted_data)

    def decrypt_file(self, input_path: str, output_path: str):
        """Decrypts a file from input_path and saves it to output_path."""
        with open(input_path, 'rb') as f_in:
            iv = f_in.read(16)
            encrypted_data = f_in.read()

        cipher = Cipher(algorithms.AES(self.key), modes.CBC(iv), backend=self.backend)
        decryptor = cipher.decryptor()
        
        padded_data = decryptor.update(encrypted_data) + decryptor.finalize()
        
        unpadder = padding.PKCS7(128).unpadder()
        data = unpadder.update(padded_data) + unpadder.finalize()

        with open(output_path, 'wb') as f_out:
            f_out.write(data)
