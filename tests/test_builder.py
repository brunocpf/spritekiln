"""Tests for the workflow builder (build/graph.py). Plain Python, no ComfyUI needed:
    python3 tests/test_builder.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "build"))
from graph import Graph  # noqa: E402

OI = {
    "LoadImage": {"input": {"required": {"image": [[], {"image_upload": True}]}},
                  "output": ["IMAGE", "MASK"], "output_name": ["IMAGE", "MASK"]},
    "SK_PixelPerfect": {"input": {"required": {
        "image": ["IMAGE"], "mode": [["fixed_size", "auto_detect", "match_reference"]],
        "width": ["INT", {"default": 64}], "seed": ["INT", {"default": 0, "control_after_generate": True}]},
        "optional": {"reference": ["IMAGE"]}},
        "output": ["IMAGE"], "output_name": ["pixel_art"]},
}


def build():
    g = Graph(OI, "t", "t")
    img = g.add("LoadImage", "A", image="x.png")
    pp = g.add("SK_PixelPerfect", "A", image=img[0], mode="match_reference", width=img[0], seed=5, reference=img[0])
    byp = g.add("SK_PixelPerfect", "A", node_mode=4, image=img[0], mode="auto_detect")
    return g, img, pp, byp


def test_inputs_named_mode_reach_the_node():
    g, img, pp, _ = build()
    api = g.to_api()
    assert api[str(pp.id)]["inputs"]["mode"] == "match_reference"
    ui = g.to_ui()
    node = next(n for n in ui["nodes"] if n["id"] == pp.id)
    assert node["mode"] == 0 and "match_reference" in node["widgets_values"]


def test_bypassed_nodes_are_dropped_from_api_and_kept_in_ui():
    g, _, _, byp = build()
    assert str(byp.id) not in g.to_api()
    assert next(n for n in g.to_ui()["nodes"] if n["id"] == byp.id)["mode"] == 4


def test_widget_order_seed_control_and_links():
    g, img, pp, _ = build()
    ui = g.to_ui()
    node = next(n for n in ui["nodes"] if n["id"] == pp.id)
    # mode, width (linked -> default kept), seed, control_after_generate
    assert node["widgets_values"] == ["match_reference", 64, 5, "randomize"]
    load = next(n for n in ui["nodes"] if n["id"] == img.id)
    assert load["widgets_values"] == ["x.png", "image"]
    linked = [i for i in node["inputs"] if i["link"] is not None]
    assert {i["name"] for i in linked} == {"image", "width", "reference"}
    assert all(l[5] == "IMAGE" for l in ui["links"] if l[3] == pp.id and l[4] != 2)


def test_unknown_input_is_rejected():
    g = Graph(OI, "t", "t")
    try:
        g.add("SK_PixelPerfect", "A", mood="typo")
    except KeyError:
        return
    raise AssertionError("typo in an input name was accepted")


if __name__ == "__main__":
    failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            try:
                fn()
                print("PASS", name)
            except AssertionError as e:
                failed += 1
                print("FAIL", name, e)
    sys.exit(1 if failed else 0)
