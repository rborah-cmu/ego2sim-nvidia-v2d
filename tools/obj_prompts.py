"""SAM2 object prompts for the evaluation episodes, read by eye off ep<EP>_grid.jpg (display coords, 1000 px per tile;
full-res = display x 2.028). Object id i = i-th object of the eval roster (track_3_evaluation_objects.json order).
Each prompt: (frame, obj_id, "box", x0, y0, x1, y1) or (frame, obj_id, "pts", [(x, y, label), ...]).
Writes ~/v2d/ego2sim/ep<EP>/obj_prompts.json on riddle when run there: python obj_prompts.py OUT_ROOT
"""
import json
import sys
from pathlib import Path

S = 2.028
P = {
    1: [  # white_pot, white_pot_lid (lid on the pot at frames 0 and 334)
        (0, 1, "box", 425, 403, 637, 590),
        (0, 0, "pts", [(465, 580, 1), (600, 575, 1), (540, 650, 1), (545, 720, 1),
                       (530, 470, 0), (400, 650, 0), (680, 620, 0), (700, 450, 0)]),
        (111, 0, "box", 340, 385, 555, 750), (111, 1, "box", 600, 412, 790, 582),
        (334, 1, "box", 400, 385, 590, 525),
        (334, 0, "pts", [(500, 660, 1), (500, 720, 1), (440, 535, 1), (575, 530, 1),
                         (495, 450, 0), (380, 650, 0), (650, 640, 0), (720, 470, 0)]),
    ],
    3: [  # white_pot, wooden_spoon
        (0, 0, "box", 325, 430, 528, 750), (0, 1, "box", 553, 427, 775, 678),
        (312, 0, "box", 280, 335, 480, 750), (312, 1, "box", 545, 395, 700, 605),
    ],
    5: [
        (0, 0, "box", 305, 370, 505, 745), (0, 1, "box", 585, 390, 750, 668),
        (287, 0, "box", 290, 300, 510, 665), (287, 1, "box", 568, 330, 750, 600),
    ],
    6: [
        (0, 0, "box", 315, 405, 505, 748), (0, 1, "box", 560, 430, 748, 693),
        (242, 0, "box", 365, 325, 560, 720), (242, 1, "box", 590, 405, 795, 665),
    ],
    8: [  # beige_cup, white_pot
        (0, 0, "box", 630, 322, 750, 500), (0, 1, "box", 375, 350, 572, 718),
        (146, 0, "box", 460, 185, 625, 410),
        (218, 0, "box", 588, 278, 700, 458), (218, 1, "box", 303, 332, 500, 700),
    ],
    10: [  # blue_cup, beige_cup (beige stacked into blue at the end)
        (0, 0, "box", 378, 512, 510, 705), (0, 1, "box", 635, 488, 780, 690),
        (53, 1, "box", 445, 265, 635, 540), (53, 0, "box", 455, 505, 575, 635),
        (160, 1, "box", 440, 440, 580, 545), (160, 0, "box", 455, 520, 575, 715),
    ],
    14: [  # blue_cup, plastic_dish_rack
        (0, 0, "pts", [(470, 620, 1), (480, 710, 1), (440, 690, 1), (600, 500, 0)]), (0, 1, "box", 318, 452, 825, 630),
        (105, 0, "box", 515, 395, 612, 560),
    ],
    15: [  # plastic_dish_rack, wooden_spoon (spoon ends upright in the rack's holder)
        (0, 0, "box", 290, 383, 795, 555), (0, 1, "box", 393, 605, 780, 688),
        (165, 0, "box", 335, 435, 832, 590),
        (165, 1, "pts", [(362, 300, 1), (372, 380, 1), (500, 520, 0)]),
    ],
    16: [  # plastic_dish_rack, wooden_spoon
        (0, 0, "box", 320, 425, 810, 595), (0, 1, "box", 408, 645, 785, 720),
        (172, 0, "box", 385, 475, 880, 615),
        (172, 1, "pts", [(420, 330, 1), (428, 420, 1), (600, 550, 0)]),
    ],
    19: [  # mini_sweeper, mini_dust_pan
        (0, 0, "box", 628, 440, 688, 598), (0, 1, "box", 380, 463, 530, 635),
        (78, 1, "box", 360, 455, 585, 610),
        (234, 0, "box", 640, 430, 720, 635), (234, 1, "box", 360, 500, 515, 700),
    ],
    20: [  # blue_cup (already in hand at frame 0), water_pitcher
        (0, 0, "box", 275, 352, 415, 535), (0, 1, "box", 405, 250, 905, 552),
        (134, 0, "box", 225, 405, 395, 590), (134, 1, "box", 535, 95, 865, 480),
        (201, 0, "box", 250, 420, 400, 595), (201, 1, "box", 540, 105, 850, 510),
    ],
    25: [  # wooden_piece_1 = LEFT plank at frame 0 (as verified on public ep23), wooden_piece_2 = right
        (0, 0, "box", 365, 425, 510, 640), (0, 1, "box", 600, 420, 795, 628),
        (118, 0, "box", 445, 325, 575, 575),
        (355, 0, "pts", [(445, 560, 1), (545, 360, 1), (415, 390, 0), (595, 495, 0)]),
        (355, 1, "pts", [(415, 390, 1), (595, 495, 1), (445, 560, 0), (545, 360, 0)]),
    ],
    32: [  # blue_cup (held at frame 0), wooden_spoon
        (0, 0, "box", 380, 335, 500, 495), (0, 1, "box", 612, 330, 782, 612),
        (172, 0, "box", 285, 290, 425, 490), (172, 1, "box", 532, 240, 755, 415),
        (258, 0, "box", 300, 335, 420, 510), (258, 1, "box", 490, 340, 750, 560),
    ],
    33: [  # blue_cup, wooden_spoon
        (0, 0, "box", 315, 368, 440, 545), (0, 1, "box", 500, 380, 755, 590),
        (116, 0, "box", 228, 290, 415, 495), (116, 1, "box", 535, 238, 950, 465),
        (347, 0, "box", 255, 310, 400, 490), (347, 1, "box", 465, 305, 725, 475),
    ],
    28: [  # plastic_dish_rack, white_pot (pot ends in the rack)
        (0, 0, "box", 240, 330, 720, 510), (0, 1, "box", 370, 410, 735, 750),
        (73, 0, "box", 185, 375, 650, 555), (73, 1, "box", 560, 95, 810, 375),
        (219, 0, "box", 215, 290, 695, 455),
        (219, 1, "pts", [(530, 340, 1), (560, 420, 1), (605, 525, 1), (330, 420, 0), (650, 400, 0)]),
    ],
    30: [  # mini_sweeper (stands in the rack at start and end), plastic_dish_rack
        (0, 0, "pts", [(543, 420, 1), (550, 500, 1), (450, 500, 0), (650, 470, 0)]),
        (0, 1, "box", 300, 390, 852, 580),
        (87, 0, "box", 690, 360, 760, 500), (87, 1, "box", 220, 370, 770, 560),
        (262, 0, "pts", [(478, 420, 1), (490, 490, 1), (400, 510, 0), (600, 480, 0)]),
        (262, 1, "box", 235, 380, 775, 590),
    ],
    35: [  # wooden_spoon
        (0, 0, "box", 403, 345, 668, 540), (280, 0, "box", 428, 395, 830, 525), (420, 0, "box", 365, 335, 660, 450),
    ],
    36: [  # wooden_spoon
        (0, 0, "box", 348, 400, 640, 520), (271, 0, "box", 343, 255, 440, 690), (406, 0, "box", 430, 410, 735, 545),
    ],
    37: [  # wooden_spoon
        (0, 0, "box", 360, 460, 690, 518), (50, 0, "box", 285, 165, 730, 238), (150, 0, "box", 350, 450, 670, 520),
    ],
    38: [  # mini_dust_pan
        (0, 0, "box", 405, 368, 585, 540), (87, 0, "box", 612, 255, 835, 440), (262, 0, "box", 495, 355, 700, 505),
    ],
}


def to_json(items):
    out = []
    for it in items:
        f, oid, kind = it[:3]
        p = {"frame_index": f, "object_id": oid, "role": "object", "points": [], "point_labels": []}
        if kind == "box":
            x0, y0, x1, y1 = (int(round(v * S)) for v in it[3:])
            p["box"] = {"x0": x0, "y0": y0, "x1": min(x1, 2027), "y1": min(y1, 1519)}
        else:
            p["points"] = [{"x": int(round(x * S)), "y": int(round(y * S))} for x, y, _ in it[3]]
            p["point_labels"] = [int(l) for _, _, l in it[3]]
        out.append(p)
    return {"prompts": out}


if __name__ == "__main__":
    root = Path(sys.argv[1])
    for ep, items in P.items():
        d = root / f"ep{ep}"
        d.mkdir(parents=True, exist_ok=True)
        json.dump(to_json(items), open(d / "obj_prompts.json", "w"))
        print(f"ep{ep}: {len(items)} prompts")
