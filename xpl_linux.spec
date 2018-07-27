# -*- mode: python -*-

block_cipher = None


a = Analysis(
    ['xpl/__main__.py'],
    pathex=['/home/simon/projects/xpl_buildwin/xpl'],
    binaries=[],
    datas=[
        ("xpl/menubar.ui", "xpl/"),
        ("xpl/rsf.db", "xpl/"),
        ("xpl/xpl.glade", "xpl/"),
        ("xpl/xpl_catalog.xml", "xpl/"),
        ("xpl/assets/atom_lib.png", "xpl/assets/"),
        ("xpl/assets/xpl.svg", "xpl/assets/"),
        ("xpl/assets/xpl48.png", "xpl/assets/"),
        ("xpl/assets/pan.png", "xpl/assets/")
    ],
    hiddenimports=['scipy._lib.messagestream'],
    hookspath=[],
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    exclude_binaries=True,
    name='XPL',
    debug=False,
    strip=False,
    upx=True,
    console=False
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    name='xpl-linux'
)
