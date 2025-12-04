import bpy
import bpy_extras
import json
from pathlib import Path
import random
import uuid
import os
import math
import time
import copy
from math import sin, cos, tan, pi

jsondir = os.path.expanduser("~/schoolwork/APS360 Deep Learning/fabscrape/cards.json")
with open(jsondir, 'r') as file:
    cards = json.load(file)

datagen_json = os.path.expanduser("~/schoolwork/APS360 Deep Learning/data_generation/cards.json")

img_dir = os.path.expanduser("~/schoolwork/APS360 Deep Learning/fabscrape/pngs/")

out_img_dir = os.path.expanduser("~/schoolwork/APS360 Deep Learning/data_generation/images")

LIGHT_DISTANCE = 20
LIGHT_Z_MIN = 5

CAMERA_DISTANCE = 30

def gen_point(distance, limits):
    # algorithm to generate random point on sphere https://mathworld.wolfram.com/SpherePointPicking.html
    while True:
        r = distance
        u = random.random()
        v = random.random()
        theta = 2 * pi * u
        phi = math.acos(2*v - 1)

        x = r * sin(phi) * cos(theta)
        y = r * sin(phi) * sin(theta)
        z = r * cos(phi)
        
        good = True
        for item, limit in zip((x,y,z), limits):
            if limit is not None:
                if (limit[0] is not None and item < limit[0]):
                    good = False
                if (limit[1] is not None and item > limit[1]):
                    good = False
        if good:
            return x, y, z

def randomize_state(card):
    bpy.data.objects['Area'].location = gen_point(LIGHT_DISTANCE, (None, None, (LIGHT_Z_MIN, None)))
    bpy.data.objects['Area.001'].location = gen_point(LIGHT_DISTANCE, (None, None, (LIGHT_Z_MIN, None)))
    bpy.data.objects['Area.002'].location = gen_point(LIGHT_DISTANCE, (None, None, (LIGHT_Z_MIN, None)))
    bpy.data.objects['Camera'].location = gen_point(CAMERA_DISTANCE, ((-3, 3), (2, 10), (15, 29)))

    bpy.data.objects['Card'].location = (random.uniform(-3, 3), random.uniform(-3, 3), 0)

    filename = card['filename'].replace('.webp', '.png')
    filepath = os.path.join(img_dir, filename)
    b_img = bpy.data.images.load(filepath)

    b_mat_node = bpy.data.materials['Material.001'].node_tree.nodes["Image Texture"]
    b_mat_node.image = b_img

    b_rot_node = bpy.data.scenes['Scene'].node_tree.nodes['Vector Rotate']
    b_rot_node.inputs["Angle"].default_value = random.randint(0,360)

    b_val_node = bpy.data.scenes['Scene'].node_tree.nodes['Value']
    b_val_node.outputs["Value"].default_value = random.randint(0,7)


    prefixes = ["CL_", "FL_", "GL_", "LI_", "Hologram", "MI_", "OSL_", "Plasma"]
    prefixes += ["Cardboard", "Resin", "Rust", "Wood_Planks", "Clay01", "Cycles_Lava_Ame", "Molten_Steel", "Wrought_Iron"]  # displacement causes card to not be seen
    
    mats = []
    lib = bpy.context.preferences.filepaths.asset_libraries.get("blend_metapack")
    library_path = Path(lib.path)
    blend_files = [fp for fp in library_path.glob("**/*.blend") if fp.is_file()]
    #https://blender.stackexchange.com/questions/244971/how-do-i-get-all-assets-in-a-given-userassetlibrary-with-the-python-api

    for blend_file in blend_files:
        with bpy.data.libraries.load(str(blend_file), assets_only=True) as (data_from, data_to):
            mats += data_from.materials
            #load, only do once
            #data_to.materials = mats

    materials = [i for i in mats if i is not None and (not any(i.startswith(prefix) for prefix in prefixes))]

    floor = bpy.data.objects['Floor']

    mat = bpy.data.materials.get(random.choice(materials))

    if not floor.data.materials:
        floor.data.materials.append(mat)
    else:
        floor.data.materials[0] = mat




def get_card_coords():

    bpy.context.view_layer.update()  # update the card location info since I just updated them

    card = bpy.data.objects['Card']
    mw = card.matrix_world
    vertex_coords = [get_final_coords(mw @ vertex.co) for vertex in card.data.vertices]

    return vertex_coords

def get_final_coords(coords):
    # https://blender.stackexchange.com/questions/882/how-to-find-image-coordinates-of-the-rendered-vertex
    scene = bpy.context.scene

    co_2d = bpy_extras.object_utils.world_to_camera_view(scene, bpy.data.objects['Camera'], coords)

    render_scale = scene.render.resolution_percentage / 100
    render_size = (
        int(scene.render.resolution_x * render_scale),
        int(scene.render.resolution_y * render_scale),
    )

    return round(co_2d.x * render_size[0]), round(co_2d.y * render_size[1])

def render():
    bpy.ops.render.render(write_still=True)

def clean():
    for img in bpy.data.images:
        if not img.users:
            bpy.data.images.remove(img)


if os.path.exists(datagen_json):
    with open(datagen_json, 'r') as file:
        data = json.load(file)
else:
    data = []

for i in range(10000):
#while True:
#for i in range(1):
    card = random.choice(cards)

    output_filename = str(uuid.uuid4()) + '.png'
    
    randomize_state(card)
    coords = get_card_coords()

    bpy.context.scene.render.filepath = os.path.join(out_img_dir, output_filename)

    render()

    carddata = {}
    carddata['filename'] = output_filename
    carddata['card_id'] = card['card_id']
    carddata['coords'] = coords
    try:
        carddata['floor_material'] = bpy.data.objects['Floor'].data.materials[0].name
    except:
        carddata['floor_material'] = 'no_material'
    b_val_node = bpy.data.scenes['Scene'].node_tree.nodes['Value']
    carddata['blur_amount'] = b_val_node.outputs["Value"].default_value

    data.append(carddata)

    with open(datagen_json, 'w') as file:
        json.dump(data, file)

    clean()
