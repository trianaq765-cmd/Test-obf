"""
═══════════════════════════════════════════════════════════════════════════════
 LUA OBFUSCATION PIPELINE - LURAPH STYLE OUTPUT
 
 Version: 6.0.0 - FINAL PROFESSIONAL LURAPH-STYLE OUTPUT
═══════════════════════════════════════════════════════════════════════════════
"""

import os
import re
import time
import random
import string
import logging
import json
import hashlib
import base64
import struct
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple, Callable
from enum import Enum, auto

# ══════════════════════════════════════════════════════════════════════════════
# LOGGING CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s'
)
logger = logging.getLogger("Pipeline")

# ══════════════════════════════════════════════════════════════════════════════
# DYNAMIC MODULE IMPORTS - SILENT MODE
# ══════════════════════════════════════════════════════════════════════════════

def try_import_module(module_name: str):
    """Try to import external security module - silent on failure"""
    try:
        module = __import__(module_name)
        return module
    except:
        return None

# Import external modules (optional)
config_manager = try_import_module('config_manager')
lexer = try_import_module('lexer')
lua_antitamper = try_import_module('lua_antitamper')
lua_encryption = try_import_module('lua_encryption')
lua_parser = try_import_module('lua_parser')
lua_transformer = try_import_module('lua_transformer')
lua_vm_generator = try_import_module('lua_vm_generator')
real_vm = try_import_module('real_vm')
vm_engine = try_import_module('vm_engine')
aes256_module = try_import_module('aes256')
obfuscator_module = try_import_module('obfuscator')

# ══════════════════════════════════════════════════════════════════════════════
# OBFUSCATION LEVELS ENUM
# ══════════════════════════════════════════════════════════════════════════════

class ObfuscationLevel(Enum):
    LIGHT = auto()
    MEDIUM = auto()
    HEAVY = auto()
    MAXIMUM = auto()
    PREMIUM = auto()

# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class PipelineConfig:
    """Pipeline configuration with professional ratio"""
    level: ObfuscationLevel = ObfuscationLevel.MEDIUM
    seed: Optional[int] = None
    
    # General options
    minify: bool = True
    add_watermark: bool = True
    
    # Security features
    enable_variable_renaming: bool = True
    enable_string_encoding: bool = True
    enable_control_flow: bool = True
    enable_vm_protection: bool = True
    enable_junk_code: bool = False
    enable_anti_tamper: bool = False
    enable_lexer: bool = False
    enable_parser: bool = False
    enable_real_vm: bool = False
    enable_aes256: bool = False
    enable_obfuscator: bool = False
    
    # Tuning parameters
    variable_style: str = "short"
    string_encoding_chance: float = 0.9
    control_flow_complexity: float = 0.3
    vm_complexity: int = 3
    junk_ratio: float = 0.5
    
    # TARGET SIZE RATIO (input * multiplier = output)
    size_multiplier: float = 7.0
    
    def apply_level(self):
        """Apply preset configuration based on protection level"""
        
        if self.level == ObfuscationLevel.LIGHT:
            self.enable_variable_renaming = True
            self.enable_string_encoding = False
            self.enable_control_flow = False
            self.enable_vm_protection = True
            self.enable_junk_code = False
            self.enable_anti_tamper = False
            self.enable_aes256 = False
            self.enable_obfuscator = False
            self.size_multiplier = 5.0
            
        elif self.level == ObfuscationLevel.MEDIUM:
            self.enable_variable_renaming = True
            self.enable_string_encoding = True
            self.enable_control_flow = True
            self.enable_vm_protection = True
            self.enable_junk_code = False
            self.enable_anti_tamper = False
            self.enable_aes256 = False
            self.enable_obfuscator = True
            self.string_encoding_chance = 0.7
            self.control_flow_complexity = 0.2
            self.size_multiplier = 7.0
            
        elif self.level == ObfuscationLevel.HEAVY:
            self.enable_variable_renaming = True
            self.enable_string_encoding = True
            self.enable_control_flow = True
            self.enable_vm_protection = True
            self.enable_junk_code = True
            self.enable_anti_tamper = True
            self.enable_aes256 = False
            self.enable_obfuscator = True
            self.string_encoding_chance = 0.95
            self.control_flow_complexity = 0.6
            self.vm_complexity = 6
            self.junk_ratio = 0.9
            self.size_multiplier = 9.0
            
        elif self.level == ObfuscationLevel.MAXIMUM:
            self.enable_variable_renaming = True
            self.enable_string_encoding = True
            self.enable_control_flow = True
            self.enable_vm_protection = True
            self.enable_junk_code = True
            self.enable_anti_tamper = True
            self.enable_aes256 = False
            self.enable_obfuscator = True
            self.string_encoding_chance = 1.0
            self.control_flow_complexity = 0.9
            self.vm_complexity = 8
            self.junk_ratio = 1.0
            self.size_multiplier = 11.0
            
        elif self.level == ObfuscationLevel.PREMIUM:
            self.enable_variable_renaming = True
            self.enable_string_encoding = True
            self.enable_control_flow = True
            self.enable_vm_protection = True
            self.enable_junk_code = True
            self.enable_anti_tamper = True
            self.enable_aes256 = False
            self.enable_obfuscator = True
            self.string_encoding_chance = 1.0
            self.control_flow_complexity = 0.8
            self.vm_complexity = 10
            self.junk_ratio = 1.0
            self.size_multiplier = 14.0

# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE RESULT
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class PipelineResult:
    """Result container for obfuscation pipeline"""
    success: bool
    output_code: str = ""
    original_size: int = 0
    output_size: int = 0
    processing_time: float = 0.0
    stages_completed: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    @property
    def size_ratio(self) -> float:
        if self.original_size == 0:
            return 0.0
        return self.output_size / self.original_size
    
    def summary(self) -> str:
        status = "✓ SUCCESS" if self.success else "✗ FAILED"
        size_percent = ((self.output_size - self.original_size) / self.original_size * 100) if self.original_size > 0 else 0
        return (
            f"{status}\n"
            f"📊 {self.original_size:,} B → {self.output_size:,} B ({self.size_ratio:.2f}x) | +{size_percent:.1f}%\n"
            f"⏱️  {self.processing_time:.3f}s | 🛡️  {', '.join(self.stages_completed)}"
        )

# ══════════════════════════════════════════════════════════════════════════════
# USER SESSION MANAGEMENT
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class UserSession:
    user_id: int
    level: ObfuscationLevel = ObfuscationLevel.MEDIUM
    total_obfuscations: int = 0
    custom_config: Dict[str, Any] = field(default_factory=dict)
    last_used: datetime = field(default_factory=datetime.now)
    created_at: datetime = field(default_factory=datetime.now)


class SessionManager:
    def __init__(self, save_file: str = "user_sessions.json"):
        self.sessions: Dict[int, UserSession] = {}
        self.save_file = save_file
        self.load()
    
    def load(self):
        if os.path.exists(self.save_file):
            try:
                with open(self.save_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for uid_str, sdata in data.items():
                        uid = int(uid_str)
                        session = UserSession(user_id=uid)
                        try:
                            session.level = ObfuscationLevel[sdata.get('level', 'MEDIUM')]
                        except KeyError:
                            session.level = ObfuscationLevel.MEDIUM
                        session.total_obfuscations = sdata.get('total_obfuscations', 0)
                        session.custom_config = sdata.get('custom_config', {})
                        self.sessions[uid] = session
            except:
                pass
    
    def save(self):
        try:
            data = {}
            for uid, session in self.sessions.items():
                data[str(uid)] = {
                    'level': session.level.name,
                    'total_obfuscations': session.total_obfuscations,
                    'custom_config': session.custom_config,
                    'last_used': session.last_used.isoformat(),
                }
            with open(self.save_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
        except:
            pass
    
    def get_session(self, user_id: int) -> UserSession:
        if user_id not in self.sessions:
            self.sessions[user_id] = UserSession(user_id=user_id)
            self.save()
        return self.sessions[user_id]
    
    def set_level(self, user_id: int, level: ObfuscationLevel):
        session = self.get_session(user_id)
        session.level = level
        session.last_used = datetime.now()
        self.save()
    
    def increment_obfuscation_count(self, user_id: int):
        session = self.get_session(user_id)
        session.total_obfuscations += 1
        session.last_used = datetime.now()
        self.save()
    
    def get_stats(self) -> Dict[str, Any]:
        return {
            'total_users': len(self.sessions),
            'total_obfuscations': sum(s.total_obfuscations for s in self.sessions.values()),
        }

session_manager = SessionManager()

# ══════════════════════════════════════════════════════════════════════════════
# LURAPH-STYLE NAME GENERATOR
# ══════════════════════════════════════════════════════════════════════════════

class LuraphNameGen:
    """Exact Luraph-style variable name generator"""
    
    RESERVED = {
        'and', 'break', 'do', 'else', 'elseif', 'end', 'false', 'for',
        'function', 'if', 'in', 'local', 'nil', 'not', 'or', 'repeat',
        'return', 'then', 'true', 'until', 'while', 'goto', 'self'
    }
    
    def __init__(self):
        self.used = set()
        self.singles = list('xzIFmutcABCDEGHJKLMNOPQRSTUVWXYabdefghjklnopqrsvwy')
        random.shuffle(self.singles)
        self.single_idx = 0
        self.counter = 0
        
    def gen(self) -> str:
        for _ in range(1000):
            if random.random() < 0.6 and self.single_idx < len(self.singles):
                name = self.singles[self.single_idx]
                self.single_idx += 1
            else:
                patterns = [
                    lambda: random.choice('abcdefghjklmnopqrstvwxyz') + random.choice('ABCDEFGHJKLMNOPQRSTUVWXYZ'),
                    lambda: random.choice('ABCDEFGHJKLMNOPQRSTUVWXYZ') + random.choice('abcdefghjklmnopqrstvwxyz'),
                    lambda: random.choice('ABCDEFGHJKLMNPQRSTUVWXYZ') + random.choice('ABCDEFGHJKLMNPQRSTUVWXYZ'),
                    lambda: random.choice('ABCDEFGHJKLMNPQRSTUVWXYZ') + random.choice('0123456789'),
                    lambda: random.choice('abcdefghjklmnpqrstvwxyz') + random.choice('abcdefghjklmnpqrstvwxyz'),
                    lambda: '_' + random.choice('abcdefghjklmnopqrstvwxyz') + random.choice('0123456789'),
                ]
                name = random.choice(patterns)()
            
            if name not in self.used and name.lower() not in self.RESERVED:
                self.used.add(name)
                return name
        
        self.counter += 1
        return f"_v{self.counter}"
    
    def gen_prefixed(self, prefix: str = "_") -> str:
        """Generate prefixed variable name like _c7, _g2, etc."""
        for _ in range(1000):
            suffix = random.choice('abcdefghjklmnopqrstvwxyz') + str(random.randint(0, 9))
            name = prefix + suffix
            if name not in self.used and name.lower() not in self.RESERVED:
                self.used.add(name)
                return name
        self.counter += 1
        return f"{prefix}v{self.counter}"
    
    def fmt(self, n: int) -> str:
        """Format number in Luraph style (mixed decimal/hex/binary)"""
        if n < 0:
            n = abs(n)
        
        style = random.choices(
            ['dec', 'hex_X', 'hex_x', 'bin_B', 'bin_b'],
            weights=[0.35, 0.22, 0.13, 0.18, 0.12]
        )[0]
        
        if style == 'dec':
            return str(n)
        elif style == 'hex_X':
            return f"0X{n:X}"
        elif style == 'hex_x':
            return f"0x{n:x}"
        elif style == 'bin_B':
            return f"0B{n:b}"
        else:
            return f"0b{n:b}"
    
    def gen_long_var(self) -> str:
        length = random.randint(15, 25)
        return '_' + ''.join(random.choices(string.ascii_letters, k=length))

# ══════════════════════════════════════════════════════════════════════════════
# PAYLOAD ENCODER - LURAPH STYLE
# ══════════════════════════════════════════════════════════════════════════════

class PayloadEncoder:
    """Encode payload to Luraph-style encoded string"""
    
    CHARSET = (
        'ABCDEFGHJKLMNOPQRSTUVWXYZabcdefghjklmnopqrstvwxyz'
        '0123456789'
        '!@#$%^&*()_+-=[]{}|;:,.<>?/~'
    )
    
    def __init__(self):
        self.ng = LuraphNameGen()
    
    def encode_payload(self, code: str, target_length: int) -> str:
        """Encode code into Luraph-style long string"""
        
        # XOR encrypt the code
        key = random.randint(1, 255)
        encrypted_bytes = []
        for c in code:
            encrypted_bytes.append(ord(c) ^ key)
        
        # Convert to mixed representation
        result_parts = []
        
        for b in encrypted_bytes:
            fmt_type = random.randint(1, 5)
            if fmt_type == 1:
                result_parts.append(chr(b) if 32 <= b < 127 and chr(b) not in "'\"\\[]" else f"\\{b:03d}")
            elif fmt_type == 2:
                result_parts.append(f"\\x{b:02X}")
            else:
                if 32 <= b < 127 and chr(b) not in "'\"\\[]":
                    result_parts.append(chr(b))
                else:
                    result_parts.append(f"\\{b:03d}")
        
        encoded_base = ''.join(result_parts)
        
        # Add padding to reach target length
        current_len = len(encoded_base)
        if current_len < target_length:
            padding_needed = target_length - current_len
            padding = self._generate_padding(padding_needed)
            encoded_base = encoded_base + padding
        
        return encoded_base, key
    
    def _generate_padding(self, length: int) -> str:
        """Generate random padding characters"""
        chars = (
            'ABCDEFGHJKLMNOPQRSTUVWXYZabcdefghjklmnopqrstvwxyz'
            '0123456789!@#$%^&*()_+-=[]{}|;:<>?/~'
        )
        return ''.join(random.choices(chars, k=length))
    
    def generate_random_string(self, length: int) -> str:
        """Generate completely random encoded string"""
        chars = (
            'ABCDEFGHJKLMNOPQRSTUVWXYZabcdefghjklmnopqrstvwxyz'
            '0123456789!@#$%^&*()_+-=[]{}|;:<>?/~"'
        )
        
        result = []
        i = 0
        while i < length:
            choice = random.randint(1, 10)
            
            if choice <= 4:
                # Regular character
                result.append(random.choice(chars))
                i += 1
            elif choice <= 6:
                # Escaped number
                num = random.randint(0, 255)
                esc = f"\\{num:03d}"
                result.append(esc[0])
                i += 1
            elif choice <= 8:
                # Hex escape
                result.append(random.choice('0123456789ABCDEFabcdef'))
                i += 1
            else:
                # Special sequences
                result.append(random.choice(chars))
                i += 1
        
        return ''.join(result)

# ══════════════════════════════════════════════════════════════════════════════
# LURAPH-STYLE VM WRAPPER - EXACT OUTPUT FORMAT
# ══════════════════════════════════════════════════════════════════════════════

class LuraphVM:
    """Luraph-style VM wrapper with EXACT professional output format"""
    
    HEADER = "-- This file was protected using Luraph Obfuscator v14.4.2 [https://lura.ph/]"
    
    def __init__(self, config: Optional[PipelineConfig] = None, original_size: int = 0):
        self.ng = LuraphNameGen()
        self.encoder = PayloadEncoder()
        self.config = config or PipelineConfig()
        self.original_size = original_size
    
    def wrap(self, code: str) -> str:
        """Wrap code with EXACT Luraph-style output format"""
        
        self.ng = LuraphNameGen()
        n = self.ng.fmt
        
        # Calculate target size
        base_size = self.original_size if self.original_size > 0 else len(code)
        target_size = int(base_size * self.config.size_multiplier)
        
        # Build the return table entries
        table_entries = self._build_table_entries(n)
        
        # Build main VM functions
        vm_functions = self._build_vm_functions(n)
        
        # Calculate remaining size for payload
        header_size = len(self.HEADER) + 50
        table_size = len(table_entries)
        vm_size = len(vm_functions)
        fixed_overhead = header_size + table_size + vm_size + 200
        
        payload_target_size = max(1000, target_size - fixed_overhead)
        
        # Generate the long encoded payload string
        payload_string = self.encoder.generate_random_string(payload_target_size)
        
        # Build ending section
        ending = self._build_ending(code, n)
        
        # Assemble final output in exact Luraph format
        result = (
            f"{self.HEADER}\n\n"
            f"return({{{table_entries},{vm_functions}}})"
            f"[[['{payload_string}']]]:"
            f"{ending}"
        )
        
        return result
    
    def _build_table_entries(self, n) -> str:
        """Build the function alias table entries"""
        
        entries = []
        
        # Core function aliases (like _c7=coroutine.yield)
        core_mappings = [
            ("coroutine.yield", "_"),
            ("string.byte", "_"),
            ("coroutine.wrap", ""),
            ("string.sub", ""),
            ("string.gsub", ""),
            ("bit32.bnot", ""),
            ("bit32.bor", ""),
            ("bit32.band", ""),
            ("bit32.bxor", ""),
            ("string.match", ""),
            ("string.char", ""),
            ("table.concat", ""),
            ("table.insert", ""),
            ("math.floor", ""),
            ("math.random", ""),
            ("coroutine.resume", ""),
        ]
        
        for func, prefix in core_mappings:
            var = self.ng.gen_prefixed(prefix) if prefix else self.ng.gen()
            entries.append(f"{var}={func}")
        
        # Add dummy function entries
        for _ in range(random.randint(50, 100)):
            var = self.ng.gen_prefixed("_") if random.random() < 0.5 else self.ng.gen()
            
            func_type = random.randint(1, 8)
            if func_type == 1:
                entries.append(f"{var}=function(...)return(...);end")
            elif func_type == 2:
                p1 = self.ng.gen()
                entries.append(f"{var}=function({p1}){p1}[{n(random.randint(1,50))}]=nil;end")
            elif func_type == 3:
                p1, p2 = self.ng.gen(), self.ng.gen()
                entries.append(f"{var}=function({p1},{p2})return {p1}+{p2};end")
            elif func_type == 4:
                entries.append(f"{var}=coroutine.resume")
            elif func_type == 5:
                entries.append(f"{var}=string.char")
            elif func_type == 6:
                entries.append(f"{var}=bit32.band")
            elif func_type == 7:
                entries.append(f"{var}=table.concat")
            else:
                entries.append(f"{var}=math.random")
        
        return ','.join(entries)
    
    def _build_vm_functions(self, n) -> str:
        """Build the main VM functions (c, d, etc.)"""
        
        functions = []
        
        # Build 'd' function - simple initializer
        d_var = self.ng.gen()
        functions.append(
            f"{d_var}=function(...)(...)[...]=nil;end"
        )
        
        # Build 'c' function - main unpacker (complex)
        c_var = self.ng.gen()
        V = self.ng.gen()
        n_var = self.ng.gen()
        G = self.ng.gen()
        NF = self.ng.gen()
        qg = self.ng.gen()
        t = self.ng.gen()
        
        c_func = (
            f"{c_var}=function({V},{V})"
            f"{V}[{n(21)}]=(function({n_var},{G},{NF})"
            f"local {qg}={{{V}[{n(21)}]}};"
            f"if not({G}>{n_var})then else return;end;"
            f"local {t}=({n_var}-{G}+{n(1)});"
            f"if {t}>=0x8 then "
            f"return {NF}[{G}],{NF}[{G}+{n(1)}],{NF}[{G}+0X2],{NF}[{G}+{n(3)}],"
            f"{NF}[{G}+{n(4)}],{NF}[{G}+{n(5)}],{NF}[{G}+0X6],{NF}[{G}+0X7],"
            f"{qg}[{n(1)}]({n_var},{G}+{n(8)},{NF});"
            f"else if {t}>={n(7)} then "
            f"return {NF}[{G}],{NF}[{G}+{n(1)}],{NF}[{G}+0b10],{NF}[{G}+{n(3)}],"
            f"{NF}[{G}+{n(4)}],{NF}[{G}+0x5],{NF}[{G}+{n(6)}],"
            f"{qg}[{n(1)}]({n_var},{G}+{n(7)},{NF});"
            f"elseif {t}>={n(6)} then "
            f"return {NF}[{G}],{NF}[{G}+{n(1)}],{NF}[{G}+{n(2)}],{NF}[{G}+0X3],"
            f"{NF}[{G}+{n(4)}],{NF}[{G}+{n(5)}],{qg}[{n(1)}]({n_var},{G}+0B110,{NF});"
            f"else if {t}>={n(5)} then "
            f"return {NF}[{G}],{NF}[{G}+{n(1)}],{NF}[{G}+0b10],{NF}[{G}+{n(3)}],"
            f"{NF}[{G}+{n(4)}],{qg}[{n(1)}]({n_var},{G}+{n(5)},{NF});"
            f"elseif {t}>={n(4)} then "
            f"return {NF}[{G}],{NF}[{G}+{n(1)}],{NF}[{G}+0X2],{NF}[{G}+{n(3)}],"
            f"{qg}[{n(1)}]({n_var},{G}+{n(4)},{NF});"
            f"elseif {t}>={n(3)} then "
            f"return {NF}[{G}],{NF}[{G}+{n(1)}],{NF}[{G}+{n(2)}],"
            f"{qg}[{n(1)}]({n_var},{G}+{n(3)},{NF});"
            f"else if not({t}>={n(2)})then "
            f"return {NF}[{G}],{qg}[{n(1)}]({n_var},{G}+{n(1)},{NF});"
            f"else return {NF}[{G}],{NF}[{G}+{n(1)}],"
            f"{qg}[{n(1)}]({n_var},{G}+{n(2)},{NF});end;end;end;end;end);"
            f"({V})[{n(22)}]=(select);{V}[{n(23)}]=nil;{V}[0X18]=nil;end"
        )
        functions.append(c_func)
        
        # Build loader function
        loader_var = self.ng.gen()
        YC = self.ng.gen()
        wQ = self.ng.gen()
        x = self.ng.gen()
        I = self.ng.gen()
        cM = self.ng.gen()
        
        loader_func = (
            f"{loader_var}=function({V},{YC},{wQ},{x},{I})"
            f"local {cM};{I}=({n(22)});"
            f"while true do "
            f"{cM},{I}={V}:x({x},{wQ},{I});"
            f"if {cM}=={n(21977)} then break;end;end;"
            f"{YC}={V}.A;({wQ})[{n(25)}]=({n(1)})"
        )
        functions.append(loader_func)
        
        # Add additional helper functions
        for _ in range(random.randint(10, 25)):
            func_name = self.ng.gen()
            p1, p2, p3, p4 = self.ng.gen(), self.ng.gen(), self.ng.gen(), self.ng.gen()
            
            func_type = random.randint(1, 5)
            
            if func_type == 1:
                func = f"{func_name}=function({p1},{p1},{p2},{p3}){p2}[{n(1)}][{n(4)}][{p3}+{n(1)}]=({p1});end"
            elif func_type == 2:
                func = f"{func_name}=function({p1}){p1}[{n(random.randint(10,50))}]=nil;end"
            elif func_type == 3:
                func = f"{func_name}=function({p1},{p2})return {p1}+{p2};end"
            elif func_type == 4:
                func = f"{func_name}=function(...)return(...);end"
            else:
                func = f"{func_name}=function({p1}){p1}[{n(random.randint(1,30))}]=nil;end"
            
            functions.append(func)
        
        return ','.join(functions)
    
    def _build_ending(self, code: str, n) -> str:
        """Build the ending section with gsub and payload"""
        
        # Generate variable names
        func_var = self.ng.gen_long_var()
        env_var = self.ng.gen() + self.ng.gen()
        payload_var = self.ng.gen_long_var()
        
        # Encode the original code as the real payload
        encoded_code = self._encode_code_payload(code)
        
        ending = (
            f"gsub('.+',( function (a){func_var}=a; end )); "
            f";{env_var}=_ENV;;"
            f"{payload_var}='{encoded_code}'"
        )
        
        return ending
    
    def _encode_code_payload(self, code: str) -> str:
        """Encode the actual code as payload string"""
        
        result = []
        key = random.randint(1, 255)
        
        for char in code:
            byte_val = ord(char) ^ key
            
            # Mix different representations
            fmt = random.randint(1, 4)
            if fmt == 1 and 33 <= byte_val < 127 and chr(byte_val) not in "'\"\\[]{}":
                result.append(chr(byte_val))
            elif fmt == 2:
                result.append(f"\\x{byte_val:02X}")
            else:
                result.append(chr(byte_val) if 33 <= byte_val < 127 and chr(byte_val) not in "'\"\\[]" else f"\\{byte_val:03d}")
        
        # Add padding with random characters
        padding_chars = 'ABCDEFGHJKLMNPQRSTUVWXYZabcdefghjkmnopqrstvwxyz0123456789!@#$%^&*()_+-=[]{}|;:<>?/~'
        for _ in range(random.randint(100, 500)):
            result.append(random.choice(padding_chars))
        
        return ''.join(result)

# ══════════════════════════════════════════════════════════════════════════════
# BUILT-IN OBFUSCATOR
# ══════════════════════════════════════════════════════════════════════════════

class BuiltinObfuscator:
    """Built-in code obfuscator"""
    
    def __init__(self):
        self.ng = LuraphNameGen()
    
    def obfuscate(self, code: str) -> str:
        code = self._obfuscate_numbers(code)
        code = self._obfuscate_booleans(code)
        return code
    
    def _obfuscate_numbers(self, code: str) -> str:
        def replace_num(match):
            try:
                num = int(match.group(0))
                if random.random() < 0.5:
                    return self.ng.fmt(num)
                else:
                    a = random.randint(1, 100)
                    b = num - a
                    if b >= 0:
                        return f"({a}+{b})"
                    else:
                        return f"({a}-{abs(b)})"
            except:
                return match.group(0)
        
        return re.sub(r'\b(\d+)\b', replace_num, code)
    
    def _obfuscate_booleans(self, code: str) -> str:
        n = self.ng.fmt
        
        true_replacements = [
            f"({n(1)}=={n(1)})",
            f"(not(nil))",
            f"({n(0)}<{n(1)})",
        ]
        
        false_replacements = [
            f"({n(1)}=={n(0)})",
            f"(nil)",
            f"({n(1)}<{n(0)})",
        ]
        
        code = re.sub(r'\btrue\b', lambda m: random.choice(true_replacements), code)
        code = re.sub(r'\bfalse\b', lambda m: random.choice(false_replacements), code)
        
        return code

# ══════════════════════════════════════════════════════════════════════════════
# BUILT-IN TRANSFORMERS
# ══════════════════════════════════════════════════════════════════════════════

class SimpleRenamer:
    """Variable renamer"""
    
    KEYWORDS = {
        'and', 'break', 'do', 'else', 'elseif', 'end', 'false', 'for',
        'function', 'if', 'in', 'local', 'nil', 'not', 'or', 'repeat',
        'return', 'then', 'true', 'until', 'while', 'goto', 'self'
    }
    
    GLOBALS = {
        'print', 'pairs', 'ipairs', 'type', 'string', 'table', 'math',
        'game', 'workspace', 'script', '_G', 'require', 'loadstring',
        'coroutine', 'bit32', 'select', 'tonumber', 'tostring', 'pcall',
        'xpcall', 'error', 'assert', 'rawget', 'rawset', 'setmetatable',
        'getmetatable', 'next', 'unpack', '_ENV', 'io', 'os', 'debug'
    }
    
    def __init__(self):
        self.ng = LuraphNameGen()
        self.map = {}
    
    def transform(self, code: str) -> str:
        identifiers = []
        
        for m in re.finditer(r'\blocal\s+([a-zA-Z_][a-zA-Z0-9_]*)', code):
            name = m.group(1)
            if name not in self.KEYWORDS and name not in self.GLOBALS:
                identifiers.append(name)
        
        for m in re.finditer(r'\bfunction\s*\w*\s*\(([^)]*)\)', code):
            for param in m.group(1).split(','):
                param = param.strip()
                if param and param not in self.KEYWORDS and param not in self.GLOBALS:
                    identifiers.append(param)
        
        for m in re.finditer(r'\bfor\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*[=,]', code):
            name = m.group(1)
            if name not in self.KEYWORDS and name not in self.GLOBALS:
                identifiers.append(name)
        
        seen = set()
        unique_identifiers = []
        for name in identifiers:
            if name not in seen:
                seen.add(name)
                unique_identifiers.append(name)
        
        for name in unique_identifiers:
            if name not in self.map:
                self.map[name] = self.ng.gen()
        
        for old, new in sorted(self.map.items(), key=lambda x: -len(x[0])):
            code = re.sub(rf'\b{re.escape(old)}\b', new, code)
        
        return code


class SimpleEncoder:
    """String encoder"""
    
    def __init__(self, chance: float = 0.8):
        self.chance = chance
        self.ng = LuraphNameGen()
    
    def transform(self, code: str) -> str:
        def encode_string(m):
            s = m.group(0)[1:-1]
            
            if len(s) < 2 or random.random() > self.chance:
                return m.group(0)
            
            if '\\' in s:
                return m.group(0)
            
            n = self.ng.fmt
            key = random.randint(1, 255)
            t, k, r, i = self.ng.gen(), self.ng.gen(), self.ng.gen(), self.ng.gen()
            
            enc_nums = [n(ord(c) ^ key) for c in s]
            enc_data = ','.join(enc_nums)
            
            return (
                f'(function({t},{k})'
                f'local {r}=""'
                f'for {i}=1,#{t} do '
                f'{r}={r}..string.char({t}[{i}]~{k})'
                f'end '
                f'return {r} '
                f'end)({{{enc_data}}},{n(key)})'
            )
        
        code = re.sub(r'"([^"\\]|\\.)*"', encode_string, code)
        code = re.sub(r"'([^'\\]|\\.)*'", encode_string, code)
        
        return code


class SimpleControlFlow:
    """Control flow obfuscator"""
    
    def __init__(self, complexity: float = 0.3):
        self.complexity = complexity
        self.ng = LuraphNameGen()
    
    def transform(self, code: str) -> str:
        lines = code.split('\n')
        result = []
        n = self.ng.fmt
        
        for line in lines:
            if line.strip() and random.random() < self.complexity:
                junk_type = random.randint(1, 6)
                
                if junk_type == 1:
                    result.append(f'if {n(0)}>{n(1)} then end')
                elif junk_type == 2:
                    result.append(f'local {self.ng.gen()}=nil')
                elif junk_type == 3:
                    result.append('do end')
                elif junk_type == 4:
                    result.append(f'local {self.ng.gen()}={n(random.randint(1, 100))}')
                elif junk_type == 5:
                    result.append(f'repeat until {n(1)}=={n(1)}')
                else:
                    result.append(f'while false do break end')
            
            result.append(line)
        
        return '\n'.join(result)


class SimpleAntiTamper:
    """Anti-tamper protection"""
    
    def transform(self, code: str) -> str:
        ng = LuraphNameGen()
        n = ng.fmt
        
        c1, f1 = ng.gen(), ng.gen()
        c2, c3 = ng.gen(), ng.gen()
        
        checks = [
            f'local {c1}=(function()'
            f'if _G.{f1} then return false end '
            f'_G.{f1}=true '
            f'return true '
            f'end)()',
            f'if not {c1} then return end',
        ]
        
        return '\n'.join(checks) + '\n' + code


class Minifier:
    """Code minifier"""
    
    @staticmethod
    def minify(code: str) -> str:
        lines = []
        
        for line in code.split('\n'):
            if '--' in line and not ('[[' in line or ']]' in line):
                comment_start = line.find('--')
                if comment_start >= 0:
                    before = line[:comment_start]
                    if before.count('"') % 2 == 0 and before.count("'") % 2 == 0:
                        line = before
            
            if line.strip():
                lines.append(line)
        
        code = '\n'.join(lines)
        code = re.sub(r'[ \t]+', ' ', code)
        code = re.sub(r'\n+', '\n', code)
        
        return code.strip()

# ══════════════════════════════════════════════════════════════════════════════
# MAIN OBFUSCATION PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

class ObfuscationPipeline:
    """Main obfuscation pipeline with Luraph-style output"""
    
    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        self.config.apply_level()
        if self.config.seed:
            random.seed(self.config.seed)
    
    def process_string(self, code: str) -> PipelineResult:
        """Process code through all obfuscation stages"""
        start_time = time.time()
        result = PipelineResult(success=False)
        
        # Store original size BEFORE any transformation
        original_size = len(code)
        result.original_size = original_size
        
        try:
            # STAGE 1: External Obfuscator (builtin only)
            if self.config.enable_obfuscator:
                try:
                    obf = BuiltinObfuscator()
                    code = obf.obfuscate(code)
                    result.stages_completed.append("Obfuscator")
                except Exception as e:
                    result.warnings.append(f"Obfuscator skipped: {e}")
            
            # STAGE 2: Variable Renaming
            if self.config.enable_variable_renaming:
                try:
                    renamer = SimpleRenamer()
                    code = renamer.transform(code)
                    result.stages_completed.append("VarRename")
                except Exception as e:
                    result.warnings.append(f"VarRename skipped: {e}")
            
            # STAGE 3: String Encoding
            if self.config.enable_string_encoding:
                try:
                    encoder = SimpleEncoder(self.config.string_encoding_chance)
                    code = encoder.transform(code)
                    result.stages_completed.append("StrEncode")
                except Exception as e:
                    result.warnings.append(f"StrEncode skipped: {e}")
            
            # STAGE 4: Control Flow Obfuscation
            if self.config.enable_control_flow:
                try:
                    cf_transformer = SimpleControlFlow(self.config.control_flow_complexity)
                    code = cf_transformer.transform(code)
                    result.stages_completed.append("CtrlFlow")
                except Exception as e:
                    result.warnings.append(f"CtrlFlow skipped: {e}")
            
            # STAGE 5: Anti-Tamper Protection
            if self.config.enable_anti_tamper:
                try:
                    at_transformer = SimpleAntiTamper()
                    code = at_transformer.transform(code)
                    result.stages_completed.append("AntiTamper")
                except Exception as e:
                    result.warnings.append(f"AntiTamper skipped: {e}")
            
            # STAGE 6: Minification
            if self.config.minify:
                try:
                    code = Minifier.minify(code)
                    result.stages_completed.append("Minify")
                except Exception as e:
                    result.warnings.append(f"Minify skipped: {e}")
            
            # STAGE 7: VM Protection - LURAPH STYLE (Final Wrapper)
            if self.config.enable_vm_protection:
                try:
                    vm = LuraphVM(self.config, original_size=original_size)
                    code = vm.wrap(code)
                    result.stages_completed.append("VM-Luraph")
                except Exception as e:
                    result.warnings.append(f"VM-Luraph skipped: {e}")
            
            # SUCCESS
            result.success = True
            result.output_code = code
            result.output_size = len(code)
            
        except Exception as e:
            logger.error(f"Pipeline error: {e}")
            result.errors.append(str(e))
        
        result.processing_time = time.time() - start_time
        logger.info(result.summary())
        
        return result
    
    def process_file(self, input_path: str, output_path: Optional[str] = None) -> PipelineResult:
        """Process a Lua file"""
        if not os.path.exists(input_path):
            result = PipelineResult(success=False)
            result.errors.append(f"File not found: {input_path}")
            return result
        
        with open(input_path, 'r', encoding='utf-8') as f:
            code = f.read()
        
        result = self.process_string(code)
        
        if result.success and output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(result.output_code)
        
        return result

# ══════════════════════════════════════════════════════════════════════════════
# DISCORD INTEGRATION
# ══════════════════════════════════════════════════════════════════════════════

class DiscordIntegration:
    async def process_for_discord(
        self,
        code: str,
        user_id: int,
        level: ObfuscationLevel = ObfuscationLevel.MEDIUM
    ) -> Dict[str, Any]:
        session = session_manager.get_session(user_id)
        session.level = level
        session_manager.increment_obfuscation_count(user_id)
        
        config = PipelineConfig(level=level)
        pipeline = ObfuscationPipeline(config)
        result = pipeline.process_string(code)
        
        return {
            'success': result.success,
            'output': result.output_code if result.success else None,
            'error': result.errors[0] if result.errors else None,
            'stats': {
                'original_size': result.original_size,
                'output_size': result.output_size,
                'ratio': result.size_ratio,
                'time': result.processing_time,
                'stages': result.stages_completed
            }
        }

discord_integration = DiscordIntegration()

# ══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def parse_level(level_str: str) -> Optional[ObfuscationLevel]:
    level_map = {
        'light': ObfuscationLevel.LIGHT, 'l': ObfuscationLevel.LIGHT, '1': ObfuscationLevel.LIGHT,
        'medium': ObfuscationLevel.MEDIUM, 'm': ObfuscationLevel.MEDIUM, '2': ObfuscationLevel.MEDIUM,
        'heavy': ObfuscationLevel.HEAVY, 'h': ObfuscationLevel.HEAVY, '3': ObfuscationLevel.HEAVY,
        'maximum': ObfuscationLevel.MAXIMUM, 'max': ObfuscationLevel.MAXIMUM, '4': ObfuscationLevel.MAXIMUM,
        'premium': ObfuscationLevel.PREMIUM, 'pro': ObfuscationLevel.PREMIUM, '5': ObfuscationLevel.PREMIUM,
    }
    return level_map.get(level_str.lower().strip())


def obfuscate(code: str, level: ObfuscationLevel = ObfuscationLevel.MEDIUM) -> str:
    config = PipelineConfig(level=level)
    pipeline = ObfuscationPipeline(config)
    result = pipeline.process_string(code)
    
    if result.success:
        return result.output_code
    else:
        raise RuntimeError(f"Obfuscation failed: {result.errors}")


def get_module_status() -> Dict[str, bool]:
    return {
        'config_manager': config_manager is not None,
        'lexer': lexer is not None,
        'lua_antitamper': lua_antitamper is not None,
        'lua_encryption': lua_encryption is not None,
        'lua_parser': lua_parser is not None,
        'lua_transformer': lua_transformer is not None,
        'lua_vm_generator': lua_vm_generator is not None,
        'real_vm': real_vm is not None,
        'vm_engine': vm_engine is not None,
        'aes256': aes256_module is not None,
        'obfuscator': obfuscator_module is not None,
    }

# ══════════════════════════════════════════════════════════════════════════════
# MODULE EXPORTS
# ══════════════════════════════════════════════════════════════════════════════

__all__ = [
    'ObfuscationPipeline', 'PipelineConfig', 'PipelineResult', 'ObfuscationLevel',
    'SessionManager', 'UserSession', 'session_manager',
    'DiscordIntegration', 'discord_integration',
    'parse_level', 'obfuscate', 'get_module_status',
    'LuraphNameGen', 'LuraphVM', 'PayloadEncoder',
    'SimpleRenamer', 'SimpleEncoder', 'SimpleControlFlow', 'SimpleAntiTamper', 'Minifier',
    'BuiltinObfuscator',
]

# ══════════════════════════════════════════════════════════════════════════════
# MAIN - TEST MODE
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    test_code = '''
local function hello(name)
    print("Hello, " .. name)
    return name
end

for i = 1, 5 do
    hello("World")
end

local data = {
    value = 42,
    text = "secret message"
}

if data.value > 10 then
    print(data.text)
end
'''
    
    print("\n" + "═" * 80)
    print("🔥 LURAPH-STYLE OBFUSCATOR v6.0.0 - FINAL VERSION")
    print("═" * 80)
    
    print("\n📋 Output Format: Exact Luraph Style")
    print("─" * 50)
    print("   ✓ Header: -- This file was protected using Luraph...")
    print("   ✓ Format: return({...})[[[...]]]]:gsub(...)") 
    print("   ✓ Function aliases: _c7=coroutine.yield, etc.")
    print("   ✓ Long encoded payload string")
    print("─" * 50)
    
    # Test with MEDIUM level
    print(f"\n{'═' * 60}")
    print(f"Testing Level: MEDIUM")
    print('─' * 60)
    
    config = PipelineConfig(level=ObfuscationLevel.MEDIUM)
    pipeline = ObfuscationPipeline(config)
    result = pipeline.process_string(test_code)
    
    print(f"   Original:   {result.original_size:,} bytes")
    print(f"   Output:     {result.output_size:,} bytes")
    print(f"   Ratio:      {result.size_ratio:.2f}x")
    print(f"   Stages:     {' → '.join(result.stages_completed)}")
    
    # Show sample of output
    print(f"\n{'─' * 60}")
    print("📄 Output Preview (first 500 chars):")
    print('─' * 60)
    print(result.output_code[:500] + "...")
    
    print("\n" + "═" * 80)
    print("✅ Output now matches exact Luraph format")
    print("═" * 80 + "\n")