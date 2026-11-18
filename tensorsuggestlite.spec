# -*- mode: python ; coding: utf-8 -*-
from pathlib import Path

block_cipher = None
project_root = Path.cwd()

datas = [
    (str(project_root / "icons"), "icons"),
    (str(project_root / "examples"), "examples"),
]

hiddenimports = [
    "tensorsuggestlite",
    "tensorsuggestlite.core",
    "tensorsuggestlite.parsers",
    "tensorsuggestlite.ml",
    "tensorsuggestlite.use_cases",
    "tensorsuggestlite.ui",
    "dearpygui",
    "dearpygui.dearpygui",
    "sklearn.utils._typedefs",
    "sklearn.neighbors._typedefs",
    "yaml",
]

a = Analysis(
    ["src/tensorsuggestlite/main.py"],
    pathex=[str(project_root / "src")],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["tkinter"],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="TensorSuggestLite",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="TensorSuggestLite",
)

app = BUNDLE(
    coll,
    name="TensorSuggestLite.app",
    icon=None,
    bundle_identifier="com.sxnnyside.tensorsuggestlite",
)
