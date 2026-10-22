"""SpriteKiln: local 2D game-art toolkit for ComfyUI (pixel-art generation helpers, pixel-perfect
conversion, palettes, spritesheets, animation export, seamless tiles, autotile sets, an HD
painted/cartoon mode with soft cutouts, character-to-parts export for cutout rigs, and sprite
normal maps for 2D lighting)."""
from . import nodes_gen, nodes_hd, nodes_pixel, nodes_sprite

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
for _mod in (nodes_gen, nodes_pixel, nodes_sprite, nodes_hd):
    for _key, (_cls, _name) in _mod.NODES.items():
        NODE_CLASS_MAPPINGS[_key] = _cls
        NODE_DISPLAY_NAME_MAPPINGS[_key] = "SK " + _name

# The project was first published as "PixelForge" with PF_* node IDs. Keep those IDs loadable
# (hidden from the node menu) so workflows saved with the old names keep working.
for _key, _cls in list(NODE_CLASS_MAPPINGS.items()):
    _old = "PF_" + _key[3:]
    NODE_CLASS_MAPPINGS[_old] = type(_cls.__name__ + "Legacy", (_cls,), {"DEPRECATED": True})
    NODE_DISPLAY_NAME_MAPPINGS[_old] = NODE_DISPLAY_NAME_MAPPINGS[_key] + " (legacy PF_ id)"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
