# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                         aes256.py                                          ║
# ║                    AES-256 Encryption for Lua Strings                      ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

from typing import List, Union
import base64
import os

class AES256:
    """
    Pure Python AES-256 Implementation
    For production, consider using: from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
    """
    
    # S-Box
    SBOX = [
        0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5, 0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab, 0x76,
        0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0, 0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4, 0x72, 0xc0,
        0xb7, 0xfd, 0x93, 0x26, 0x36, 0x3f, 0xf7, 0xcc, 0x34, 0xa5, 0xe5, 0xf1, 0x71, 0xd8, 0x31, 0x15,
        0x04, 0xc7, 0x23, 0xc3, 0x18, 0x96, 0x05, 0x9a, 0x07, 0x12, 0x80, 0xe2, 0xeb, 0x27, 0xb2, 0x75,
        0x09, 0x83, 0x2c, 0x1a, 0x1b, 0x6e, 0x5a, 0xa0, 0x52, 0x3b, 0xd6, 0xb3, 0x29, 0xe3, 0x2f, 0x84,
        0x53, 0xd1, 0x00, 0xed, 0x20, 0xfc, 0xb1, 0x5b, 0x6a, 0xcb, 0xbe, 0x39, 0x4a, 0x4c, 0x58, 0xcf,
        0xd0, 0xef, 0xaa, 0xfb, 0x43, 0x4d, 0x33, 0x85, 0x45, 0xf9, 0x02, 0x7f, 0x50, 0x3c, 0x9f, 0xa8,
        0x51, 0xa3, 0x40, 0x8f, 0x92, 0x9d, 0x38, 0xf5, 0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff, 0xf3, 0xd2,
        0xcd, 0x0c, 0x13, 0xec, 0x5f, 0x97, 0x44, 0x17, 0xc4, 0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19, 0x73,
        0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a, 0x90, 0x88, 0x46, 0xee, 0xb8, 0x14, 0xde, 0x5e, 0x0b, 0xdb,
        0xe0, 0x32, 0x3a, 0x0a, 0x49, 0x06, 0x24, 0x5c, 0xc2, 0xd3, 0xac, 0x62, 0x91, 0x95, 0xe4, 0x79,
        0xe7, 0xc8, 0x37, 0x6d, 0x8d, 0xd5, 0x4e, 0xa9, 0x6c, 0x56, 0xf4, 0xea, 0x65, 0x7a, 0xae, 0x08,
        0xba, 0x78, 0x25, 0x2e, 0x1c, 0xa6, 0xb4, 0xc6, 0xe8, 0xdd, 0x74, 0x1f, 0x4b, 0xbd, 0x8b, 0x8a,
        0x70, 0x3e, 0xb5, 0x66, 0x48, 0x03, 0xf6, 0x0e, 0x61, 0x35, 0x57, 0xb9, 0x86, 0xc1, 0x1d, 0x9e,
        0xe1, 0xf8, 0x98, 0x11, 0x69, 0xd9, 0x8e, 0x94, 0x9b, 0x1e, 0x87, 0xe9, 0xce, 0x55, 0x28, 0xdf,
        0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68, 0x41, 0x99, 0x2d, 0x0f, 0xb0, 0x54, 0xbb, 0x16,
    ]
    
    # Inverse S-Box
    SBOX_INV = [
        0x52, 0x09, 0x6a, 0xd5, 0x30, 0x36, 0xa5, 0x38, 0xbf, 0x40, 0xa3, 0x9e, 0x81, 0xf3, 0xd7, 0xfb,
        0x7c, 0xe3, 0x39, 0x82, 0x9b, 0x2f, 0xff, 0x87, 0x34, 0x8e, 0x43, 0x44, 0xc4, 0xde, 0xe9, 0xcb,
        0x54, 0x7b, 0x94, 0x32, 0xa6, 0xc2, 0x23, 0x3d, 0xee, 0x4c, 0x95, 0x0b, 0x42, 0xfa, 0xc3, 0x4e,
        0x08, 0x2e, 0xa1, 0x66, 0x28, 0xd9, 0x24, 0xb2, 0x76, 0x5b, 0xa2, 0x49, 0x6d, 0x8b, 0xd1, 0x25,
        0x72, 0xf8, 0xf6, 0x64, 0x86, 0x68, 0x98, 0x16, 0xd4, 0xa4, 0x5c, 0xcc, 0x5d, 0x65, 0xb6, 0x92,
        0x6c, 0x70, 0x48, 0x50, 0xfd, 0xed, 0xb9, 0xda, 0x5e, 0x15, 0x46, 0x57, 0xa7, 0x8d, 0x9d, 0x84,
        0x90, 0xd8, 0xab, 0x00, 0x8c, 0xbc, 0xd3, 0x0a, 0xf7, 0xe4, 0x58, 0x05, 0xb8, 0xb3, 0x45, 0x06,
        0xd0, 0x2c, 0x1e, 0x8f, 0xca, 0x3f, 0x0f, 0x02, 0xc1, 0xaf, 0xbd, 0x03, 0x01, 0x13, 0x8a, 0x6b,
        0x3a, 0x91, 0x11, 0x41, 0x4f, 0x67, 0xdc, 0xea, 0x97, 0xf2, 0xcf, 0xce, 0xf0, 0xb4, 0xe6, 0x73,
        0x96, 0xac, 0x74, 0x22, 0xe7, 0xad, 0x35, 0x85, 0xe2, 0xf9, 0x37, 0xe8, 0x1c, 0x75, 0xdf, 0x6e,
        0x47, 0xf1, 0x1a, 0x71, 0x1d, 0x29, 0xc5, 0x89, 0x6f, 0xb7, 0x62, 0x0e, 0xaa, 0x18, 0xbe, 0x1b,
        0xfc, 0x56, 0x3e, 0x4b, 0xc6, 0xd2, 0x79, 0x20, 0x9a, 0xdb, 0xc0, 0xfe, 0x78, 0xcd, 0x5a, 0xf4,
        0x1f, 0xdd, 0xa8, 0x33, 0x88, 0x07, 0xc7, 0x31, 0xb1, 0x12, 0x10, 0x59, 0x27, 0x80, 0xec, 0x5f,
        0x60, 0x51, 0x7f, 0xa9, 0x19, 0xb5, 0x4a, 0x0d, 0x2d, 0xe5, 0x7a, 0x9f, 0x93, 0xc9, 0x9c, 0xef,
        0xa0, 0xe0, 0x3b, 0x4d, 0xae, 0x2a, 0xf5, 0xb0, 0xc8, 0xeb, 0xbb, 0x3c, 0x83, 0x53, 0x99, 0x61,
        0x17, 0x2b, 0x04, 0x7e, 0xba, 0x77, 0xd6, 0x26, 0xe1, 0x69, 0x14, 0x63, 0x55, 0x21, 0x0c, 0x7d,
    ]
    
    # Round constants
    RCON = [0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1b, 0x36]
    
    def __init__(self, key: Union[str, bytes]):
        """Initialize AES-256 with a 32-byte key"""
        if isinstance(key, str):
            key = key.encode('utf-8')
        
        # Pad or truncate key to 32 bytes
        if len(key) < 32:
            key = key + b'\x00' * (32 - len(key))
        elif len(key) > 32:
            key = key[:32]
        
        self.key = list(key)
        self.expanded_key = self._key_expansion()
    
    @staticmethod
    def _gmul(a: int, b: int) -> int:
        """Galois Field multiplication"""
        p = 0
        for _ in range(8):
            if b & 1:
                p ^= a
            hi_bit = a & 0x80
            a = (a << 1) & 0xFF
            if hi_bit:
                a ^= 0x1b
            b >>= 1
        return p
    
    def _key_expansion(self) -> List[List[int]]:
        """Expand the key for all rounds"""
        Nk = 8  # 256-bit = 8 words
        Nr = 14  # 14 rounds for AES-256
        Nb = 4  # Block size in words
        
        W = []
        
        # Copy original key
        for i in range(Nk):
            W.append([
                self.key[i * 4],
                self.key[i * 4 + 1],
                self.key[i * 4 + 2],
                self.key[i * 4 + 3]
            ])
        
        # Expand key
        for i in range(Nk, Nb * (Nr + 1)):
            temp = W[i - 1].copy()
            
            if i % Nk == 0:
                # RotWord
                temp = temp[1:] + temp[:1]
                # SubWord
                temp = [self.SBOX[b] for b in temp]
                # XOR with Rcon
                temp[0] ^= self.RCON[i // Nk - 1]
            elif Nk > 6 and i % Nk == 4:
                # SubWord (additional for AES-256)
                temp = [self.SBOX[b] for b in temp]
            
            W.append([W[i - Nk][j] ^ temp[j] for j in range(4)])
        
        return W
    
    def _sub_bytes(self, state: List[List[int]]) -> None:
        """Apply S-Box substitution"""
        for i in range(4):
            for j in range(4):
                state[i][j] = self.SBOX[state[i][j]]
    
    def _sub_bytes_inv(self, state: List[List[int]]) -> None:
        """Apply inverse S-Box substitution"""
        for i in range(4):
            for j in range(4):
                state[i][j] = self.SBOX_INV[state[i][j]]
    
    def _shift_rows(self, state: List[List[int]]) -> None:
        """Shift rows transformation"""
        state[1] = state[1][1:] + state[1][:1]
        state[2] = state[2][2:] + state[2][:2]
        state[3] = state[3][3:] + state[3][:3]
    
    def _shift_rows_inv(self, state: List[List[int]]) -> None:
        """Inverse shift rows transformation"""
        state[1] = state[1][-1:] + state[1][:-1]
        state[2] = state[2][-2:] + state[2][:-2]
        state[3] = state[3][-3:] + state[3][:-3]
    
    def _mix_columns(self, state: List[List[int]]) -> None:
        """Mix columns transformation"""
        for i in range(4):
            a = [state[j][i] for j in range(4)]
            state[0][i] = self._gmul(a[0], 2) ^ self._gmul(a[1], 3) ^ a[2] ^ a[3]
            state[1][i] = a[0] ^ self._gmul(a[1], 2) ^ self._gmul(a[2], 3) ^ a[3]
            state[2][i] = a[0] ^ a[1] ^ self._gmul(a[2], 2) ^ self._gmul(a[3], 3)
            state[3][i] = self._gmul(a[0], 3) ^ a[1] ^ a[2] ^ self._gmul(a[3], 2)
    
    def _mix_columns_inv(self, state: List[List[int]]) -> None:
        """Inverse mix columns transformation"""
        for i in range(4):
            a = [state[j][i] for j in range(4)]
            state[0][i] = self._gmul(a[0], 14) ^ self._gmul(a[1], 11) ^ self._gmul(a[2], 13) ^ self._gmul(a[3], 9)
            state[1][i] = self._gmul(a[0], 9) ^ self._gmul(a[1], 14) ^ self._gmul(a[2], 11) ^ self._gmul(a[3], 13)
            state[2][i] = self._gmul(a[0], 13) ^ self._gmul(a[1], 9) ^ self._gmul(a[2], 14) ^ self._gmul(a[3], 11)
            state[3][i] = self._gmul(a[0], 11) ^ self._gmul(a[1], 13) ^ self._gmul(a[2], 9) ^ self._gmul(a[3], 14)
    
    def _add_round_key(self, state: List[List[int]], round: int) -> None:
        """XOR state with round key"""
        for i in range(4):
            for j in range(4):
                state[i][j] ^= self.expanded_key[round * 4 + j][i]
    
    def _encrypt_block(self, block: List[int]) -> List[int]:
        """Encrypt a single 16-byte block"""
        Nr = 14
        
        # Create state matrix (column-major)
        state = [[block[j * 4 + i] for j in range(4)] for i in range(4)]
        
        # Initial round
        self._add_round_key(state, 0)
        
        # Main rounds
        for round in range(1, Nr):
            self._sub_bytes(state)
            self._shift_rows(state)
            self._mix_columns(state)
            self._add_round_key(state, round)
        
        # Final round
        self._sub_bytes(state)
        self._shift_rows(state)
        self._add_round_key(state, Nr)
        
        # Convert back to bytes
        result = []
        for j in range(4):
            for i in range(4):
                result.append(state[i][j])
        
        return result
    
    def _decrypt_block(self, block: List[int]) -> List[int]:
        """Decrypt a single 16-byte block"""
        Nr = 14
        
        # Create state matrix
        state = [[block[j * 4 + i] for j in range(4)] for i in range(4)]
        
        # Initial round
        self._add_round_key(state, Nr)
        
        # Main rounds
        for round in range(Nr - 1, 0, -1):
            self._shift_rows_inv(state)
            self._sub_bytes_inv(state)
            self._add_round_key(state, round)
            self._mix_columns_inv(state)
        
        # Final round
        self._shift_rows_inv(state)
        self._sub_bytes_inv(state)
        self._add_round_key(state, 0)
        
        # Convert back to bytes
        result = []
        for j in range(4):
            for i in range(4):
                result.append(state[i][j])
        
        return result
    
    def encrypt(self, plaintext: Union[str, bytes]) -> bytes:
        """Encrypt plaintext using AES-256-CBC"""
        if isinstance(plaintext, str):
            plaintext = plaintext.encode('utf-8')
        
        data = list(plaintext)
        
        # PKCS7 Padding
        pad_len = 16 - (len(data) % 16)
        data.extend([pad_len] * pad_len)
        
        # Generate random IV
        iv = list(os.urandom(16))
        
        # CBC mode encryption
        result = iv.copy()
        prev_block = iv
        
        for i in range(0, len(data), 16):
            block = data[i:i + 16]
            # XOR with previous ciphertext block
            block = [block[j] ^ prev_block[j] for j in range(16)]
            encrypted = self._encrypt_block(block)
            result.extend(encrypted)
            prev_block = encrypted
        
        return bytes(result)
    
    def decrypt(self, ciphertext: bytes) -> bytes:
        """Decrypt ciphertext using AES-256-CBC"""
        data = list(ciphertext)
        
        # Extract IV
        iv = data[:16]
        data = data[16:]
        
        # CBC mode decryption
        result = []
        prev_block = iv
        
        for i in range(0, len(data), 16):
            block = data[i:i + 16]
            decrypted = self._decrypt_block(block)
            # XOR with previous ciphertext block
            decrypted = [decrypted[j] ^ prev_block[j] for j in range(16)]
            result.extend(decrypted)
            prev_block = block
        
        # Remove PKCS7 padding
        if result:
            pad_len = result[-1]
            if 0 < pad_len <= 16:
                result = result[:-pad_len]
        
        return bytes(result)
    
    def encrypt_to_base64(self, plaintext: str) -> str:
        """Encrypt and return base64 encoded string"""
        encrypted = self.encrypt(plaintext)
        return base64.b64encode(encrypted).decode('ascii')
    
    def decrypt_from_base64(self, ciphertext: str) -> str:
        """Decrypt base64 encoded ciphertext"""
        encrypted = base64.b64decode(ciphertext)
        decrypted = self.decrypt(encrypted)
        return decrypted.decode('utf-8')


# ══════════════════════════════════════════════════════════════════════════════
# ALTERNATIVE: Using cryptography library (Recommended for production)
# ══════════════════════════════════════════════════════════════════════════════

class AES256Fast:
    """
    AES-256 using cryptography library (faster and more secure)
    Install: pip install cryptography
    """
    
    def __init__(self, key: Union[str, bytes]):
        try:
            from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
            from cryptography.hazmat.backends import default_backend
            self._crypto_available = True
        except ImportError:
            self._crypto_available = False
            print("Warning: cryptography library not found, using pure Python AES")
            self._fallback = AES256(key)
            return
        
        if isinstance(key, str):
            key = key.encode('utf-8')
        
        # Pad or truncate key to 32 bytes
        if len(key) < 32:
            key = key + b'\x00' * (32 - len(key))
        elif len(key) > 32:
            key = key[:32]
        
        self.key = key
        self._backend = default_backend()
        self._algorithms = algorithms
        self._modes = modes
        self._Cipher = Cipher
    
    def encrypt(self, plaintext: Union[str, bytes]) -> bytes:
        if not self._crypto_available:
            return self._fallback.encrypt(plaintext)
        
        if isinstance(plaintext, str):
            plaintext = plaintext.encode('utf-8')
        
        # PKCS7 Padding
        pad_len = 16 - (len(plaintext) % 16)
        plaintext = plaintext + bytes([pad_len] * pad_len)
        
        # Generate random IV
        iv = os.urandom(16)
        
        cipher = self._Cipher(
            self._algorithms.AES(self.key),
            self._modes.CBC(iv),
            backend=self._backend
        )
        encryptor = cipher.encryptor()
        ciphertext = encryptor.update(plaintext) + encryptor.finalize()
        
        return iv + ciphertext
    
    def decrypt(self, ciphertext: bytes) -> bytes:
        if not self._crypto_available:
            return self._fallback.decrypt(ciphertext)
        
        iv = ciphertext[:16]
        ciphertext = ciphertext[16:]
        
        cipher = self._Cipher(
            self._algorithms.AES(self.key),
            self._modes.CBC(iv),
            backend=self._backend
        )
        decryptor = cipher.decryptor()
        plaintext = decryptor.update(ciphertext) + decryptor.finalize()
        
        # Remove PKCS7 padding
        pad_len = plaintext[-1]
        return plaintext[:-pad_len]
    
    def encrypt_to_base64(self, plaintext: str) -> str:
        encrypted = self.encrypt(plaintext)
        return base64.b64encode(encrypted).decode('ascii')
    
    def decrypt_from_base64(self, ciphertext: str) -> str:
        encrypted = base64.b64decode(ciphertext)
        decrypted = self.decrypt(encrypted)
        return decrypted.decode('utf-8')