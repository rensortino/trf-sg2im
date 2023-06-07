import PIL
from einops import rearrange
import torch
import cv2
from utils.data import inv_normalize, imagenet_preprocess
from utils.vis import draw_scene_graph as original_draw_sg
COLORS = [
            (137, 49, 239), # Blue-Violet
            (242, 202, 25), # Yellow
            (255, 0, 189), # Pink
            (0, 87, 233), # Blue
            (135, 233, 17), # Green
            (225, 24, 69), # Orange
            (1, 176, 231), # Cyan
            (138, 10, 183), # Violet
            (138, 59, 59), # Brown
        ]

HEXCOLORS = [
                '#8931EF', # Blue-Violet
                '#F2CA19', # Yellow
                '#FF00BD', # Pink
                '#0057E9', # Blue
                '#87E911', # Green
                '#FF1845', # Orange
                '#01B0E7', # Cyan
                '#8A0AB7', # Violet
                '#8A3B3B', # Brown
            ]

def preprocess_image(img):
    img_w_boxes = img.clone()
    img_w_boxes = inv_normalize(img_w_boxes) * 255
    img_w_boxes = rearrange(img_w_boxes, 'c h w -> h w c')
    return img_w_boxes.cpu().numpy().copy()

def postprocess_image(img_w_boxes):
    img_w_boxes = torch.tensor(img_w_boxes)
    img_w_boxes = rearrange(img_w_boxes, 'h w c -> c h w')
    img_w_boxes = imagenet_preprocess()(img_w_boxes)
    pil_img = img_to_PIL(img_w_boxes)
    return pil_img

def draw_box(img, box_coords, idx, category=None):
    '''
    img: shape [H,W,C]
    '''

    H,W = img.shape[:2]

    # Rescale [0,1] coordinates to image size
    box_coords = box_coords * torch.tensor([W,H,W,H], device=box_coords)
    x0, y0, x1, y1 = box_coords.int().tolist()
    color = COLORS[idx % len(COLORS)]
    thickness = 1
    img_w_box = cv2.rectangle(img, (x0, y0), (x1, y1), color, thickness)
    if category:
        cv2.putText(img_w_box, category, (x0, y0+20), fontScale=0.5, color=color, thickness=1)

    # cv2.imwrite('boxes.png', img_w_box)

    return img_w_box

def remove_special_nodes(g, idx=0):
    '''
    Removes special __PAD__ and __image__ nodes with the respective
    relationships, mainly for clean visualization purposes
    '''
    batch_size = g.batch.max().item() + 1
    # Every batch has the same number of nodes (whether actual nodes or PAD)
    max_n_objs = g.batch.shape[0] // batch_size
    # Take all the indices of the selected batch, which are not padding
    keep_nodes = torch.logical_and(g.batch.eq(idx), g.y.ne(173))
    # and are not __image__
    keep_nodes = torch.logical_and(keep_nodes, g.y.ne(0))
    g.x = g.x[keep_nodes]
    g.y = g.y[keep_nodes]
    # Keep all edge indices within the selected graph index
    selected_edges = g.edge_batch == idx
    # and which are not referring to the __image__ node
    edges_with_actual_nodes = g.edge_index[1] < g.edge_index[:,selected_edges].max()
    keep_edges = torch.logical_and(selected_edges, edges_with_actual_nodes)
    g.edge_index = g.edge_index[:,keep_edges]
    if idx != 0:
        g.edge_index = g.edge_index % (max_n_objs * idx)
    g.edge_attr = g.edge_attr[keep_edges]
    return g

def get_edge2label(edges, relationships, rel_vocab):
    edge_to_label = {}
    for edge, rel in zip(edges.t(), relationships):
        edge_to_label[tuple(edge.tolist())] = rel_vocab[rel.item()]
    return edge_to_label

def draw_scene_graph(graph, vocab):
    s, o = graph.edges()
    p = graph.edata['feat']
    triples = torch.stack((s,p,o), dim=1)
    sg_img = original_draw_sg(graph.ndata['feat'], triples, vocab)
    return PIL.Image.fromarray(sg_img)
    # return torch.from_numpy(sg_img)

def draw_boxes(image, boxes):
    img_w_boxes = preprocess_image(image)

    for i, box in enumerate(boxes):
        img_w_boxes = draw_box(img_w_boxes, box, i)

    return postprocess_image(img_w_boxes)