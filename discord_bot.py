"""
═══════════════════════════════════════════════════════════════════════════════
 LUA OBFUSCATOR DISCORD BOT - FULLY INTEGRATED WITH PIPELINE
 Version: 5.3.0 - Matched with Pipeline.py
═══════════════════════════════════════════════════════════════════════════════
"""

import os
import io
import re
import asyncio
import logging
from datetime import datetime
from typing import Optional

import discord
from discord import app_commands
from discord.ext import commands

# ══════════════════════════════════════════════════════════════════════════════
# IMPORT FROM PIPELINE (Single Source of Truth)
# ══════════════════════════════════════════════════════════════════════════════

from pipeline import (
    # Core Classes
    ObfuscationPipeline,
    PipelineConfig,
    PipelineResult,
    ObfuscationLevel,
    
    # Session Management (Persistent)
    SessionManager,
    UserSession,
    session_manager,
    
    # Helpers
    parse_level,
    get_module_status,
)

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

BOT_TOKEN = os.getenv("DISCORD_BOT_TOKEN", "YOUR_BOT_TOKEN_HERE")
BOT_PREFIX = os.getenv("BOT_PREFIX", "!")
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB (increased for large files)
MAX_CODE_LENGTH = 100000         # 100KB
ALLOWED_EXTENSIONS = {'.lua', '.txt', '.luac'}

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger("Bot")

# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS - MATCHED WITH PIPELINE LEVELS
# ══════════════════════════════════════════════════════════════════════════════

LEVEL_EMOJI = {
    ObfuscationLevel.LIGHT: "🟢",
    ObfuscationLevel.MEDIUM: "🟡", 
    ObfuscationLevel.HEAVY: "🟠",
    ObfuscationLevel.MAXIMUM: "🔴",
    ObfuscationLevel.PREMIUM: "💎",  # ✅ Added PREMIUM
}

LEVEL_INFO = {
    ObfuscationLevel.LIGHT: "Basic protection (~3.5x size)",
    ObfuscationLevel.MEDIUM: "Balanced protection (~5.5x size)",
    ObfuscationLevel.HEAVY: "Strong VM + AES256 (~7.5x size)",
    ObfuscationLevel.MAXIMUM: "Maximum security (~9x size)",
    ObfuscationLevel.PREMIUM: "💎 Luraph-style ultimate! (~11x size)",  # ✅ Added
}

# Colors
COLOR_SUCCESS = 0x00FF00
COLOR_ERROR = 0xFF0000
COLOR_INFO = 0x00BFFF
COLOR_PROCESS = 0x9B59B6
COLOR_PREMIUM = 0xFFD700  # Gold for premium

# ══════════════════════════════════════════════════════════════════════════════
# EMBED BUILDER
# ══════════════════════════════════════════════════════════════════════════════

class Embed:
    """Simple embed factory"""
    
    @staticmethod
    def success(title: str, desc: str = "") -> discord.Embed:
        return discord.Embed(
            title=f"✅ {title}", 
            description=desc,
            color=COLOR_SUCCESS,
            timestamp=datetime.now()
        )
    
    @staticmethod
    def error(title: str, desc: str = "") -> discord.Embed:
        return discord.Embed(
            title=f"❌ {title}",
            description=desc, 
            color=COLOR_ERROR,
            timestamp=datetime.now()
        )
    
    @staticmethod
    def info(title: str, desc: str = "") -> discord.Embed:
        return discord.Embed(
            title=f"ℹ️ {title}",
            description=desc,
            color=COLOR_INFO,
            timestamp=datetime.now()
        )
    
    @staticmethod
    def processing() -> discord.Embed:
        return discord.Embed(
            title="⏳ Processing...",
            description="Obfuscating your code with Luraph-style protection...",
            color=COLOR_PROCESS
        )
    
    @staticmethod
    def result(r: PipelineResult, author: discord.User = None, level: ObfuscationLevel = None) -> discord.Embed:
        """Build result embed from PipelineResult"""
        if r.success:
            color = COLOR_PREMIUM if level == ObfuscationLevel.PREMIUM else COLOR_SUCCESS
            embed = discord.Embed(
                title="✅ Obfuscation Complete!",
                color=color,
                timestamp=datetime.now()
            )
            
            # Size info with percentage
            size_increase = ((r.output_size - r.original_size) / r.original_size * 100) if r.original_size > 0 else 0
            
            embed.add_field(
                name="📊 Statistics",
                value=f"```\n"
                      f"Original:  {r.original_size:,} bytes\n"
                      f"Output:    {r.output_size:,} bytes\n"
                      f"Ratio:     {r.size_ratio:.2f}x (+{size_increase:.1f}%)\n"
                      f"Time:      {r.processing_time:.3f}s\n```",
                inline=False
            )
            embed.add_field(
                name="🔧 Stages Applied",
                value=f"`{' → '.join(r.stages_completed)}`",
                inline=False
            )
            
            if level:
                embed.add_field(
                    name="🛡️ Protection Level",
                    value=f"{LEVEL_EMOJI.get(level, '⚪')} **{level.name}**",
                    inline=True
                )
            
            if r.warnings:
                embed.add_field(
                    name="⚠️ Warnings",
                    value="\n".join(f"• {w}" for w in r.warnings[:3]),
                    inline=False
                )
        else:
            embed = discord.Embed(
                title="❌ Obfuscation Failed",
                description="\n".join(r.errors) if r.errors else "Unknown error",
                color=COLOR_ERROR,
                timestamp=datetime.now()
            )
        
        if author:
            embed.set_author(name=author.display_name, icon_url=author.display_avatar.url)
        
        embed.set_footer(text="Lua Shield v5.3.0 | Luraph-style Protection")
        
        return embed

# ══════════════════════════════════════════════════════════════════════════════
# BOT SETUP
# ══════════════════════════════════════════════════════════════════════════════

intents = discord.Intents.default()
intents.message_content = True

class LuaBot(commands.Bot):
    def __init__(self):
        super().__init__(
            command_prefix=commands.when_mentioned_or(BOT_PREFIX),
            intents=intents,
            help_command=None
        )
        self.start_time = datetime.now()
    
    async def setup_hook(self):
        await self.tree.sync()
        logger.info("✓ Slash commands synced")
    
    async def on_ready(self):
        logger.info(f"✓ Ready as {self.user} | {len(self.guilds)} servers")
        
        # Log module status
        modules = get_module_status()
        loaded = sum(1 for v in modules.values() if v)
        logger.info(f"✓ Modules loaded: {loaded}/{len(modules)}")
        
        await self.change_presence(
            activity=discord.Activity(
                type=discord.ActivityType.watching,
                name=f"{BOT_PREFIX}help | Lua Shield"
            )
        )

bot = LuaBot()

# ══════════════════════════════════════════════════════════════════════════════
# CORE FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def extract_code(content: str) -> Optional[str]:
    """Extract Lua code from message"""
    # Code block with lua
    match = re.search(r'```(?:lua)?\s*([\s\S]*?)```', content)
    if match:
        return match.group(1).strip()
    
    # Inline code
    match = re.search(r'`([^`]+)`', content)
    if match:
        return match.group(1).strip()
    
    # After command prefix
    for cmd in ['obfuscate', 'obf', 'o', 'protect']:
        prefix = f'{BOT_PREFIX}{cmd}'
        if content.lower().startswith(prefix):
            code = content[len(prefix):].strip()
            if code:
                return code
    
    return None

async def do_obfuscate(code: str, user_id: int, override_level: ObfuscationLevel = None) -> tuple[PipelineResult, ObfuscationLevel]:
    """Run obfuscation with user's saved settings"""
    session = session_manager.get_session(user_id)
    
    # Use override level or session level
    level = override_level or session.level
    
    # Build config from level
    config = PipelineConfig(level=level)
    
    # Apply custom config if any
    for key, value in session.custom_config.items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    config.apply_level()
    
    # Run pipeline
    pipeline = ObfuscationPipeline(config)
    result = pipeline.process_string(code)
    
    # Update stats
    if result.success:
        session_manager.increment_obfuscation_count(user_id)
    
    return result, level

def make_file(code: str, name: str = "obfuscated.lua") -> discord.File:
    """Create Discord file from code"""
    return discord.File(io.BytesIO(code.encode('utf-8')), filename=name)

# ══════════════════════════════════════════════════════════════════════════════
# PREFIX COMMANDS
# ══════════════════════════════════════════════════════════════════════════════

@bot.command(name='help', aliases=['h', '?'])
async def cmd_help(ctx: commands.Context):
    """Show help"""
    session = session_manager.get_session(ctx.author.id)
    
    embed = discord.Embed(
        title="🔒 Lua Shield - Luraph-Style Obfuscator",
        description="Protect your Lua scripts with professional-grade obfuscation!",
        color=COLOR_INFO
    )
    
    embed.add_field(
        name="📝 Commands",
        value=f"""
`{BOT_PREFIX}obf <code>` - Obfuscate code
`{BOT_PREFIX}level [level]` - View/set level
`{BOT_PREFIX}settings` - View your settings
`{BOT_PREFIX}modules` - Show loaded modules
`{BOT_PREFIX}stats` - Bot statistics
`{BOT_PREFIX}reset` - Reset to defaults
        """,
        inline=False
    )
    
    embed.add_field(
        name="📊 Protection Levels",
        value="\n".join(
            f"{LEVEL_EMOJI[l]} `{l.name.lower()}` - {LEVEL_INFO[l]}" 
            for l in ObfuscationLevel
        ),
        inline=False
    )
    
    embed.add_field(
        name="💡 Tips",
        value="""
• Upload `.lua` files directly to obfuscate
• Use code blocks for better formatting
• Your level setting is saved permanently
• Higher levels = more protection + larger output
        """,
        inline=False
    )
    
    embed.set_footer(text=f"Your level: {LEVEL_EMOJI[session.level]} {session.level.name}")
    embed.set_author(name=ctx.author.display_name, icon_url=ctx.author.display_avatar.url)
    embed.set_thumbnail(url=bot.user.display_avatar.url)
    
    await ctx.send(embed=embed)

@bot.command(name='obfuscate', aliases=['obf', 'o', 'protect'])
async def cmd_obfuscate(ctx: commands.Context, *, code: str = None):
    """Obfuscate Lua code"""
    
    # Check for file attachment
    if ctx.message.attachments:
        await handle_file(ctx, ctx.message.attachments[0])
        return
    
    # Extract code from message
    if code is None:
        code = extract_code(ctx.message.content)
    
    if not code:
        session = session_manager.get_session(ctx.author.id)
        embed = Embed.error(
            "No Code Provided",
            f"Please provide code or attach a `.lua` file.\n\n"
            f"**Examples:**\n"
            f"```\n{BOT_PREFIX}obf print('Hello')\n```\n"
            f"```\n{BOT_PREFIX}obf\n```lua\nlocal x = 1\nprint(x)\n```\n```\n"
            f"Current level: {LEVEL_EMOJI[session.level]} **{session.level.name}**"
        )
        await ctx.send(embed=embed)
        return
    
    # Check length
    if len(code) > MAX_CODE_LENGTH:
        embed = Embed.error(
            "Code Too Long", 
            f"Maximum {MAX_CODE_LENGTH:,} characters.\n"
            f"Your code: {len(code):,} characters.\n\n"
            f"Please upload as a file instead."
        )
        await ctx.send(embed=embed)
        return
    
    # Process
    msg = await ctx.send(embed=Embed.processing())
    
    try:
        result, level = await do_obfuscate(code, ctx.author.id)
        embed = Embed.result(result, ctx.author, level)
        
        if result.success:
            if len(result.output_code) > 1900:
                file = make_file(result.output_code)
                await msg.edit(embed=embed, attachments=[file])
            else:
                embed.add_field(
                    name="📄 Output",
                    value=f"```lua\n{result.output_code[:1800]}{'...' if len(result.output_code) > 1800 else ''}\n```",
                    inline=False
                )
                await msg.edit(embed=embed)
        else:
            await msg.edit(embed=embed)
    
    except Exception as e:
        logger.error(f"Obfuscation error: {e}", exc_info=True)
        await msg.edit(embed=Embed.error("Error", str(e)[:500]))

async def handle_file(ctx: commands.Context, attachment: discord.Attachment):
    """Handle file obfuscation"""
    
    ext = os.path.splitext(attachment.filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        embed = Embed.error(
            "Invalid File Type", 
            f"Allowed extensions: {', '.join(ALLOWED_EXTENSIONS)}\n"
            f"Your file: `{ext}`"
        )
        await ctx.send(embed=embed)
        return
    
    if attachment.size > MAX_FILE_SIZE:
        embed = Embed.error(
            "File Too Large", 
            f"Maximum size: {MAX_FILE_SIZE // (1024*1024)}MB\n"
            f"Your file: {attachment.size / (1024*1024):.2f}MB"
        )
        await ctx.send(embed=embed)
        return
    
    msg = await ctx.send(embed=Embed.processing())
    
    try:
        content = await attachment.read()
        code = content.decode('utf-8')
        
        result, level = await do_obfuscate(code, ctx.author.id)
        embed = Embed.result(result, ctx.author, level)
        embed.add_field(name="📁 Source File", value=f"`{attachment.filename}` ({attachment.size:,} bytes)", inline=True)
        
        if result.success:
            out_name = f"obf_{attachment.filename}"
            if not out_name.endswith('.lua'):
                out_name = out_name.rsplit('.', 1)[0] + '.lua'
            file = make_file(result.output_code, out_name)
            await msg.edit(embed=embed, attachments=[file])
        else:
            await msg.edit(embed=embed)
    
    except UnicodeDecodeError:
        await msg.edit(embed=Embed.error("Encoding Error", "File must be UTF-8 encoded text."))
    except Exception as e:
        logger.error(f"File processing error: {e}", exc_info=True)
        await msg.edit(embed=Embed.error("Error", str(e)[:500]))

@bot.command(name='level', aliases=['lvl', 'l'])
async def cmd_level(ctx: commands.Context, level: str = None):
    """Set obfuscation level (persistent)"""
    
    session = session_manager.get_session(ctx.author.id)
    
    if level is None:
        # Show current level
        embed = Embed.info(
            "Current Level",
            f"Your level: {LEVEL_EMOJI[session.level]} **{session.level.name}**\n"
            f"{LEVEL_INFO[session.level]}\n\n"
            f"**Available levels:**\n" +
            "\n".join(f"{LEVEL_EMOJI[l]} `{l.name.lower()}` - {LEVEL_INFO[l]}" for l in ObfuscationLevel) +
            f"\n\n`{BOT_PREFIX}level <name>` to change"
        )
        await ctx.send(embed=embed)
        return
    
    new_level = parse_level(level)
    if new_level is None:
        embed = Embed.error(
            "Invalid Level", 
            f"Available levels: `light`, `medium`, `heavy`, `maximum`, `premium`\n\n"
            f"Example: `{BOT_PREFIX}level premium`"
        )
        await ctx.send(embed=embed)
        return
    
    # Save persistently
    session_manager.set_level(ctx.author.id, new_level)
    
    embed = Embed.success(
        "Level Saved!",
        f"{LEVEL_EMOJI[new_level]} **{new_level.name}**\n"
        f"{LEVEL_INFO[new_level]}\n\n"
        f"✅ This setting is saved permanently for your account!"
    )
    await ctx.send(embed=embed)

@bot.command(name='settings', aliases=['cfg', 'config'])
async def cmd_settings(ctx: commands.Context):
    """Show user settings"""
    
    session = session_manager.get_session(ctx.author.id)
    
    # Build config preview
    config = PipelineConfig(level=session.level)
    for k, v in session.custom_config.items():
        if hasattr(config, k):
            setattr(config, k, v)
    config.apply_level()
    
    embed = discord.Embed(
        title="⚙️ Your Settings",
        color=COLOR_INFO,
        timestamp=datetime.now()
    )
    
    embed.add_field(
        name="Level",
        value=f"{LEVEL_EMOJI[session.level]} {session.level.name}",
        inline=True
    )
    embed.add_field(
        name="Total Obfuscations",
        value=str(session.total_obfuscations),
        inline=True
    )
    embed.add_field(
        name="Size Multiplier",
        value=f"{config.size_multiplier}x",
        inline=True
    )
    
    # Features based on actual config attributes
    features = []
    if getattr(config, 'enable_variable_renaming', False): features.append("✅ Variable Renaming")
    if getattr(config, 'enable_string_encoding', False): features.append("✅ String Encoding")
    if getattr(config, 'enable_control_flow', False): features.append("✅ Control Flow")
    if getattr(config, 'enable_vm_protection', False): features.append("✅ VM Protection")
    if getattr(config, 'enable_aes256', False): features.append("✅ AES256 Encryption")
    if getattr(config, 'enable_obfuscator', False): features.append("✅ External Obfuscator")
    if getattr(config, 'enable_anti_tamper', False): features.append("✅ Anti-Tamper")
    if getattr(config, 'enable_real_vm', False): features.append("✅ Real VM")
    if getattr(config, 'minify', False): features.append("✅ Minify")
    
    embed.add_field(
        name="🔧 Active Features",
        value="\n".join(features) if features else "None configured",
        inline=False
    )
    
    embed.add_field(
        name="Last Used",
        value=session.last_used.strftime("%Y-%m-%d %H:%M UTC"),
        inline=True
    )
    
    embed.set_author(name=ctx.author.display_name, icon_url=ctx.author.display_avatar.url)
    embed.set_footer(text=f"{BOT_PREFIX}level <name> to change level")
    
    await ctx.send(embed=embed)

@bot.command(name='modules', aliases=['mods'])
async def cmd_modules(ctx: commands.Context):
    """Show loaded modules"""
    
    modules = get_module_status()
    
    embed = discord.Embed(
        title="📦 Pipeline Modules",
        color=COLOR_INFO,
        timestamp=datetime.now()
    )
    
    loaded = []
    not_loaded = []
    
    for name, status in modules.items():
        if status:
            loaded.append(f"✅ `{name}`")
        else:
            not_loaded.append(f"❌ `{name}`")
    
    if loaded:
        embed.add_field(name="Loaded", value="\n".join(loaded), inline=True)
    if not_loaded:
        embed.add_field(name="Not Loaded", value="\n".join(not_loaded), inline=True)
    
    embed.add_field(
        name="📊 Summary",
        value=f"**{len(loaded)}** / {len(modules)} modules loaded",
        inline=False
    )
    
    await ctx.send(embed=embed)

@bot.command(name='stats', aliases=['info', 'status'])
async def cmd_stats(ctx: commands.Context):
    """Show bot statistics"""
    
    uptime = datetime.now() - bot.start_time
    h, rem = divmod(int(uptime.total_seconds()), 3600)
    m, s = divmod(rem, 60)
    
    stats = session_manager.get_stats()
    modules = get_module_status()
    
    embed = discord.Embed(
        title="📊 Lua Shield Statistics",
        color=COLOR_INFO,
        timestamp=datetime.now()
    )
    
    embed.add_field(name="🕐 Uptime", value=f"{h}h {m}m {s}s", inline=True)
    embed.add_field(name="🌐 Servers", value=str(len(bot.guilds)), inline=True)
    embed.add_field(name="🏓 Latency", value=f"{bot.latency*1000:.0f}ms", inline=True)
    embed.add_field(name="👥 Users", value=str(stats['total_users']), inline=True)
    embed.add_field(name="🔒 Total Obfuscations", value=str(stats['total_obfuscations']), inline=True)
    embed.add_field(name="📦 Modules", value=f"{sum(modules.values())}/{len(modules)}", inline=True)
    
    # Level distribution
    if stats['level_distribution']:
        dist_lines = []
        for level_name, count in stats['level_distribution'].items():
            try:
                level = ObfuscationLevel[level_name]
                emoji = LEVEL_EMOJI.get(level, '⚪')
                dist_lines.append(f"{emoji} {level_name}: {count}")
            except KeyError:
                dist_lines.append(f"⚪ {level_name}: {count}")
        
        embed.add_field(name="📈 Level Usage", value="\n".join(dist_lines), inline=False)
    
    embed.set_thumbnail(url=bot.user.display_avatar.url)
    embed.set_footer(text=f"discord.py v{discord.__version__}")
    
    await ctx.send(embed=embed)

@bot.command(name='reset')
async def cmd_reset(ctx: commands.Context):
    """Reset settings to default"""
    session = session_manager.get_session(ctx.author.id)
    session.level = ObfuscationLevel.MEDIUM
    session.custom_config = {}
    session_manager.save()
    
    embed = Embed.success(
        "Settings Reset", 
        f"Level: {LEVEL_EMOJI[ObfuscationLevel.MEDIUM]} **MEDIUM**\n"
        f"Custom settings cleared."
    )
    await ctx.send(embed=embed)

# ══════════════════════════════════════════════════════════════════════════════
# SLASH COMMANDS
# ══════════════════════════════════════════════════════════════════════════════

@bot.tree.command(name="obfuscate", description="Obfuscate Lua code")
@app_commands.describe(
    code="Lua code to obfuscate",
    level="Override protection level for this obfuscation"
)
@app_commands.choices(level=[
    app_commands.Choice(name="🟢 Light - Basic", value="light"),
    app_commands.Choice(name="🟡 Medium - Balanced", value="medium"),
    app_commands.Choice(name="🟠 Heavy - Strong", value="heavy"),
    app_commands.Choice(name="🔴 Maximum - Extreme", value="maximum"),
    app_commands.Choice(name="💎 Premium - Luraph-style", value="premium"),
])
async def slash_obfuscate(interaction: discord.Interaction, code: str, level: str = None):
    await interaction.response.defer(thinking=True)
    
    try:
        override = parse_level(level) if level else None
        result, used_level = await do_obfuscate(code, interaction.user.id, override)
        embed = Embed.result(result, interaction.user, used_level)
        
        if result.success and len(result.output_code) > 1900:
            file = make_file(result.output_code)
            await interaction.followup.send(embed=embed, file=file)
        elif result.success:
            embed.add_field(
                name="📄 Output",
                value=f"```lua\n{result.output_code[:1800]}{'...' if len(result.output_code) > 1800 else ''}\n```",
                inline=False
            )
            await interaction.followup.send(embed=embed)
        else:
            await interaction.followup.send(embed=embed)
    except Exception as e:
        logger.error(f"Slash obfuscate error: {e}", exc_info=True)
        await interaction.followup.send(embed=Embed.error("Error", str(e)[:500]))

@bot.tree.command(name="level", description="Set your obfuscation level")
@app_commands.describe(level="Protection level")
@app_commands.choices(level=[
    app_commands.Choice(name="🟢 Light - Basic protection", value="light"),
    app_commands.Choice(name="🟡 Medium - Balanced", value="medium"),
    app_commands.Choice(name="🟠 Heavy - Strong + AES256", value="heavy"),
    app_commands.Choice(name="🔴 Maximum - Extreme security", value="maximum"),
    app_commands.Choice(name="💎 Premium - Luraph-style ultimate", value="premium"),
])
async def slash_level(interaction: discord.Interaction, level: str):
    new_level = parse_level(level) or ObfuscationLevel.MEDIUM
    session_manager.set_level(interaction.user.id, new_level)
    
    embed = Embed.success(
        "Level Saved!",
        f"{LEVEL_EMOJI[new_level]} **{new_level.name}**\n{LEVEL_INFO[new_level]}"
    )
    await interaction.response.send_message(embed=embed)

@bot.tree.command(name="settings", description="View your settings")
async def slash_settings(interaction: discord.Interaction):
    session = session_manager.get_session(interaction.user.id)
    config = PipelineConfig(level=session.level)
    config.apply_level()
    
    embed = discord.Embed(title="⚙️ Your Settings", color=COLOR_INFO)
    embed.add_field(name="Level", value=f"{LEVEL_EMOJI[session.level]} {session.level.name}", inline=True)
    embed.add_field(name="Obfuscations", value=str(session.total_obfuscations), inline=True)
    embed.add_field(name="Size Multiplier", value=f"{config.size_multiplier}x", inline=True)
    embed.set_author(name=interaction.user.display_name, icon_url=interaction.user.display_avatar.url)
    
    await interaction.response.send_message(embed=embed)

@bot.tree.command(name="help", description="Show help information")
async def slash_help(interaction: discord.Interaction):
    session = session_manager.get_session(interaction.user.id)
    
    embed = discord.Embed(
        title="🔒 Lua Shield - Help",
        description="Professional Luraph-style Lua obfuscation!",
        color=COLOR_INFO
    )
    embed.add_field(
        name="Commands",
        value=(
            "`/obfuscate` - Obfuscate code\n"
            "`/level` - Set protection level\n"
            "`/settings` - View your settings"
        ),
        inline=False
    )
    embed.add_field(
        name="Levels",
        value="\n".join(f"{LEVEL_EMOJI[l]} `{l.name.lower()}`" for l in ObfuscationLevel),
        inline=False
    )
    embed.set_footer(text=f"Your level: {LEVEL_EMOJI[session.level]} {session.level.name}")
    
    await interaction.response.send_message(embed=embed)

@bot.tree.command(name="modules", description="Show loaded pipeline modules")
async def slash_modules(interaction: discord.Interaction):
    modules = get_module_status()
    
    loaded = [f"✅ `{k}`" for k, v in modules.items() if v]
    not_loaded = [f"❌ `{k}`" for k, v in modules.items() if not v]
    
    embed = discord.Embed(title="📦 Pipeline Modules", color=COLOR_INFO)
    if loaded:
        embed.add_field(name="Loaded", value="\n".join(loaded), inline=True)
    if not_loaded:
        embed.add_field(name="Not Loaded", value="\n".join(not_loaded), inline=True)
    embed.set_footer(text=f"{len(loaded)}/{len(modules)} modules active")
    
    await interaction.response.send_message(embed=embed)

# ══════════════════════════════════════════════════════════════════════════════
# EVENT HANDLERS
# ══════════════════════════════════════════════════════════════════════════════

@bot.event
async def on_message(message: discord.Message):
    if message.author.bot:
        return
    
    # Auto-detect .lua files
    if message.attachments and not message.content.startswith(BOT_PREFIX):
        for att in message.attachments:
            if os.path.splitext(att.filename)[1].lower() in ALLOWED_EXTENSIONS:
                session = session_manager.get_session(message.author.id)
                embed = Embed.info(
                    "Lua File Detected",
                    f"📁 `{att.filename}` ({att.size:,} bytes)\n\n"
                    f"React ✅ to obfuscate with **{session.level.name}** level!\n"
                    f"React ❌ to cancel."
                )
                msg = await message.channel.send(embed=embed)
                await msg.add_reaction("✅")
                await msg.add_reaction("❌")
                
                def check(r, u):
                    return (
                        u == message.author and 
                        str(r.emoji) in ["✅", "❌"] and 
                        r.message.id == msg.id
                    )
                
                try:
                    reaction, _ = await bot.wait_for('reaction_add', timeout=60.0, check=check)
                    if str(reaction.emoji) == "✅":
                        ctx = await bot.get_context(message)
                        await msg.delete()
                        await handle_file(ctx, att)
                    else:
                        await msg.delete()
                except asyncio.TimeoutError:
                    try:
                        await msg.delete()
                    except:
                        pass
                break
    
    await bot.process_commands(message)

@bot.event
async def on_command_error(ctx: commands.Context, error: commands.CommandError):
    if isinstance(error, commands.CommandNotFound):
        return
    
    if isinstance(error, commands.MissingRequiredArgument):
        embed = Embed.error("Missing Argument", f"Required parameter: `{error.param.name}`")
        await ctx.send(embed=embed)
        return
    
    if isinstance(error, commands.CommandOnCooldown):
        embed = Embed.error("Cooldown", f"Try again in {error.retry_after:.1f}s")
        await ctx.send(embed=embed)
        return
    
    logger.error(f"Command error: {error}", exc_info=True)
    embed = Embed.error("Error", str(error)[:500])
    await ctx.send(embed=embed)

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    if BOT_TOKEN == "YOUR_BOT_TOKEN_HERE":
        print("\n" + "=" * 60)
        print("❌ ERROR: Bot token not configured!")
        print("=" * 60)
        print("\nSet the DISCORD_BOT_TOKEN environment variable:")
        print("  export DISCORD_BOT_TOKEN='your_token_here'")
        print("\nOr edit this file and replace YOUR_BOT_TOKEN_HERE")
        print("=" * 60 + "\n")
        return
    
    print("\n" + "=" * 60)
    print("🔒 Lua Shield Bot v5.3.0")
    print("=" * 60)
    print(f"Prefix: {BOT_PREFIX}")
    print(f"Max file size: {MAX_FILE_SIZE // (1024*1024)}MB")
    print("=" * 60 + "\n")
    
    logger.info("Starting Lua Shield Bot...")
    bot.run(BOT_TOKEN, log_handler=None)

if __name__ == "__main__":
    main()
