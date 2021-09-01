# coding=utf-8
# Copyleft 2019 Project LXRT

import sys
import csv
import base64
import time
import pickle

import numpy as np

import re
import torch

csv.field_size_limit(sys.maxsize)
FIELDNAMES = ["img_id", "img_h", "img_w", "objects_id", "objects_conf",
              "attrs_id", "attrs_conf", "num_boxes", "boxes", "features"]

def load_obj_tsv(fname, topk=None):
    """Load object features from tsv file.

    :param fname: The path to the tsv file.
    :param topk: Only load features for top K images (lines) in the tsv file.
        Will load all the features if topk is either -1 or None.
    :return: A list of image object features where each feature is a dict.
        See FILENAMES above for the keys in the feature dict.
    """
    data = []
    start_time = time.time()
    print("Start to load Faster-RCNN detected objects from %s" % fname)
    with open(fname) as f:
        reader = csv.DictReader(f, FIELDNAMES, delimiter="\t")
        for i, item in enumerate(reader):

            for key in ['img_h', 'img_w', 'num_boxes']:
                item[key] = int(item[key])

            boxes = item['num_boxes']
            decode_config = [
                ('objects_id', (boxes,), np.int64),
                ('objects_conf', (boxes,), np.float32),
                ('attrs_id', (boxes,), np.int64),
                ('attrs_conf', (boxes,), np.float32),
                ('boxes', (boxes, 4), np.float32),
                ('features', (boxes, -1), np.float32),
            ]
            for key, shape, dtype in decode_config:
                item[key] = np.frombuffer(base64.b64decode(item[key]), dtype=dtype)
                item[key] = item[key].reshape(shape)
                item[key].setflags(write=False)

            data.append(item)
            if topk is not None and len(data) == topk:
                break
    elapsed_time = time.time() - start_time
    print("Loaded %d images in file %s in %d seconds." % (len(data), fname, elapsed_time))
    return data


def load_obj_pkl(fname, topk=None):
    data = []
    start_time = time.time()
    print("Start to load Faster-RCNN detected objects from %s" % fname)
    with open(fname, 'rb') as f:
        img_feat = pickle.load(f)
        for i, item in enumerate(img_feat):
            item['img_h'] = int(item['image_h'])
            item['img_w'] = int(item['image_w'])
            item['num_boxes'] = int(item['num_bbox'])

            boxes = item['num_boxes']
            if boxes != 36:
                continue

            decode_config = [
                ('objects_id', (boxes,), np.int64),
                ('objects_conf', (boxes,), np.float32),
                ('attrs_id', (boxes,), np.int64),
                ('attrs_conf', (boxes,), np.float32)
            ]
            for key, shape, dtype in decode_config:
                # item[key] = np.frombuffer(base64.b64decode(item['info'].item().get(key)), dtype=dtype)
                item[key] = item['info'].item().get(key)
                item[key] = item[key].reshape(shape)
                item[key].setflags(write=False)

            # item['boxes'] = np.frombuffer(base64.b64decode(item['bbox']), dtype=np.float32)
            item['boxes'] = item['bbox']
            item['boxes'] = item['boxes'].reshape((boxes, 4))
            item['boxes'].setflags(write=False)

            # item['features'] = np.frombuffer(base64.b64decode(item['x']), dtype=np.float32)
            item['features'] = item['x']
            item['features'] = item['features'].reshape((boxes, -1))
            item['features'].setflags(write=False)

            item['img_id'] = item['info'].item().get('image_id') + ".jpg"

            data.append(item)
            if topk is not None and len(data) == topk:
                break
        elapsed_time = time.time() - start_time
        print("Loaded %d images in file %s in %d seconds." % (len(data), fname, elapsed_time))
        return data


def prepare_questions(annotations):
    """ Filter, Normalize and Tokenize question. """

    prepared = []
    questions = [q['question'] for q in annotations]

    for question in questions:
        # lower case
        question = question.lower()

        # define desired replacements here
        punctuation_dict = {'.': ' ', "'": '', '?': ' ', '_': ' ', '-': ' ', '/': ' ', ',': ' '}
        conversational_dict = {"thank you": '', "thanks": '', "thank": '', "please": '', "hello": '',
                               "hi ": ' ', "hey ": ' ', "good morning": '', "good afternoon": '', "have a nice day": '',
                               "okay": '', "goodbye": ''}

        rep = punctuation_dict
        rep.update(conversational_dict)

        # use these three lines to do the replacement
        rep = dict((re.escape(k), v) for k, v in rep.items())
        pattern = re.compile("|".join(rep.keys()))
        question = pattern.sub(lambda m: rep[re.escape(m.group(0))], question)

        # sentence to list
        question = question.split(' ')

        # remove empty strings
        question = list(filter(None, question))

        prepared.append(question)

    return prepared


def prepare_answers(annotations):
    answers = [[a['answer'] for a in ans_dict['answers']] for ans_dict in annotations]
    prepared = []

    for sample_answers in answers:
        prepared_sample_answers = []
        for answer in sample_answers:
            # lower case
            answer = answer.lower()

            # define desired replacements here
            punctuation_dict = {'.': ' ', "'": '', '?': ' ', '_': ' ', '-': ' ', '/': ' ', ',': ' '}

            rep = punctuation_dict
            rep = dict((re.escape(k), v) for k, v in rep.items())
            pattern = re.compile("|".join(rep.keys()))
            answer = pattern.sub(lambda m: rep[re.escape(m.group(0))], answer)
            prepared_sample_answers.append(answer)

        prepared.append(prepared_sample_answers)
    return prepared


def encode_question(question, token_to_index, max_length):
    question_vec = torch.zeros(max_length).long()
    length = min(len(question), max_length)
    for i in range(length):
        token = question[i]
        index = token_to_index.get(token, 0)
        question_vec[i] = index
    # empty encoded questions are a problem when packed,
    # if we set min length 1 we feed a 0 token to the RNN
    # that is not a problem since the token 0 does not represent a word
    return question_vec, max(length, 1)


def encode_answers(answers, answer_to_index):
    colors = ['absolute zero', 'acid green', 'aero', 'aero blue', 'african violet', 'air superiority blue', 'alabaster',
              'alice blue', 'alizarin', 'alloy orange', 'almond', 'amaranth', 'amaranth deep purple', 'amaranth pink',
              'amaranth purple', 'amaranth red', 'amazon', 'amber', 'Amber ', 'SAE/ECE', 'amethyst', 'android green',
              'antique brass', 'antique bronze', 'antique fuchsia', 'antique ruby', 'antique white', 'Ao ', 'English',
              'apple green', 'apricot', 'aqua', 'aquamarine', 'arctic lime', 'army green', 'artichoke',
              'arylide yellow', 'ash grey', 'asparagus', 'atomic tangerine', 'auburn', 'aureolin', 'avocado', 'azure',
              'Azure ', 'X11/web color', 'baby blue', 'baby blue eyes', 'baby pink', 'baby powder', 'baker-miller pink',
              'banana mania', 'banana yellow', 'barbie pink', 'barn red', 'battleship grey', 'beau blue', 'beaver',
              'beige', "b'dazzled blue", 'big dip o’ruby', 'big foot feet', 'bisque', 'bistre', 'bistre brown',
              'bitter lemon', 'bittersweet', 'bittersweet shimmer', 'black', 'black bean', 'black coral',
              'black leather jacket', 'black olive', 'black shadows', 'blanched almond', 'blast-off bronze',
              'bleu de france', 'blizzard blue', 'blood red', 'blue', 'Blue ', 'Crayola', 'Blue ', 'Munsell', 'Blue ',
              'NCS', 'Blue ', 'Pantone', 'Blue ', 'pigment', 'blue bell', 'blue-gray', 'blue-green', 'blue jeans',
              'blue sapphire', 'blue-violet', 'blue yonder', 'blueberry', 'blush', 'bole', 'bone', 'booger buster',
              'bottle green', 'brick red', 'bright green', 'bright lilac', 'bright maroon', 'bright navy blue',
              'bright pink', 'bright turquoise', 'Bright yellow ', 'Crayola', 'brilliant rose', 'brink pink',
              'british racing green', 'bronze', 'brown', 'Brown ', 'web', 'brown sugar', 'brunswick green', 'bud green',
              'buff', 'burgundy', 'burlywood', 'burnished brown', 'burnt orange', 'burnt sienna', 'burnt umber',
              'byzantine', 'byzantium', 'cadet', 'cadet blue', 'cadet grey', 'cadmium green', 'cadmium orange',
              'cadmium red', 'cadmium yellow', 'café au lait', 'café noir', 'calypso', 'cambridge blue', 'camel',
              'cameo pink', 'canary yellow', 'candy apple red', 'candy pink', 'caput mortuum', 'cardinal',
              'caribbean green', 'carmine', 'carnation pink', 'carnelian', 'carolina blue', 'carrot orange',
              'castleton green', 'catawba', 'cedar chest', 'celadon', 'celadon green', 'celeste', 'cerise', 'cerulean',
              'cerulean blue', 'cerulean frost', 'Cerulean ', 'Crayola', 'champagne', 'champagne pink', 'charcoal',
              'charleston green', 'charm pink', 'Chartreuse ', 'traditional', 'Chartreuse ', 'web',
              'cherry blossom pink', 'chestnut', 'china pink', 'china rose', 'chinese red', 'chinese violet',
              'Chocolate ', 'traditional', 'Chocolate ', 'web', 'cinereous', 'cinnamon satin', 'citrine', 'citron',
              'claret', 'cobalt blue', 'cocoa brown', 'coconut', 'coffee', 'columbia blue', 'congo pink', 'cool grey',
              'copper', 'copper penny', 'copper red', 'copper rose', 'coquelicot', 'coral', 'coral pink', 'cordovan',
              'cornell red', 'Cornflower blue ', 'web', 'Cornflower blue ', 'Crayola', 'cornsilk', 'cosmic cobalt',
              'cosmic latte', 'coyote brown', 'cotton candy', 'cream', 'crimson', 'Crimson ', 'UA', 'cultured', 'cyan',
              'cyber grape', 'cyber yellow', 'cyclamen', 'dandelion', 'dark blue', 'dark blue-gray', 'dark brown',
              'dark byzantium', 'dark cyan', 'dark electric blue', 'dark fuchsia', 'dark goldenrod', 'Dark gray ',
              'X11', 'dark green', 'Dark green ', 'X11', 'dark jungle green', 'dark khaki', 'dark lava', 'dark liver',
              'dark magenta', 'dark midnight blue', 'dark moss green', 'dark olive green', 'dark orange', 'dark orchid',
              'dark pastel green', 'dark purple', 'dark raspberry', 'dark red', 'dark salmon', 'dark sea green',
              'dark sienna', 'dark sky blue', 'dark slate blue', 'dark slate gray', 'dark spring green', 'dark tan',
              'dark turquoise', 'dark vanilla', 'dark violet', 'dartmouth green', "davy's grey", 'deep cerise',
              'deep champagne', 'deep chestnut', 'deep fuchsia', 'deep jungle green', 'deep lemon', 'deep mauve',
              'deep pink', 'deep sky blue', 'deep space sparkle', 'deep taupe', 'denim', 'denim blue', 'desert',
              'desert sand', 'diamond', 'dim gray', 'dingy dungeon', 'dirt', 'dodger blue', 'dogwood rose', 'duke blue',
              'dutch white', 'earth yellow', 'ebony', 'ecru', 'eerie black', 'eggplant', 'eggshell', 'egyptian blue',
              'electric blue', 'electric crimson', 'electric indigo', 'electric lime', 'electric purple',
              'electric violet', 'emerald', 'eminence', 'english lavender', 'english red', 'english vermillion',
              'english violet', 'eton blue', 'eucalyptus', 'fallow', 'falu red', 'fandango', 'fandango pink',
              'fashion fuchsia', 'fawn', 'feldgrau', 'fern green', 'field drab', 'fiery rose', 'firebrick',
              'fire engine red', 'fire opal', 'flame', 'flax', 'flirt', 'floral white', 'fluorescent blue',
              'Forest green ', 'Crayola', 'Forest green ', 'traditional', 'Forest green ', 'web', 'french beige',
              'french bistre', 'french blue', 'french fuchsia', 'french lilac', 'french lime', 'french mauve',
              'french pink', 'french raspberry', 'french rose', 'french sky blue', 'french violet', 'frostbite',
              'fuchsia', 'Fuchsia ', 'Crayola', 'fuchsia purple', 'fuchsia rose', 'fulvous', 'fuzzy wuzzy', 'gainsboro',
              'gamboge', 'garnet', 'generic viridian', 'ghost white', 'glaucous', 'glossy grape', 'go green', 'gold',
              'Gold ', 'metallic', 'Gold (web) ', 'Golden', 'Gold ', 'Crayola', 'gold fusion', 'golden brown',
              'golden poppy', 'golden yellow', 'goldenrod', 'granite gray', 'granny smith apple', 'Gray ', 'web',
              'Gray ', 'X11', 'green', 'Green ', 'Crayola', 'Green ', 'HTML/CSS color', 'Green ', 'Munsell', 'Green ',
              'NCS', 'Green ', 'Pantone', 'Green ', 'pigment', 'Green ', 'RYB', 'green-blue', 'Green-blue ', 'Crayola',
              'green-cyan', 'green lizard', 'green sheen', 'green-yellow', 'Green-yellow ', 'Crayola', 'grullo',
              'gunmetal', 'han blue', 'han purple', 'hansa yellow', 'harlequin', 'harvest gold', 'heat wave',
              'heliotrope', 'heliotrope gray', 'hollywood cerise', 'honeydew', 'honolulu blue', "hooker's green",
              'hot fuchsia', 'hot magenta', 'hot pink', 'hunter green', 'iceberg', 'icterine', 'illuminating emerald',
              'imperial red', 'inchworm', 'independence', 'india green', 'indian red', 'indian yellow', 'indigo',
              'indigo dye', 'International orange ', 'aerospace', 'International orange ', 'engineering',
              'International orange ', 'Golden Gate Bridge', 'iris', 'irresistible', 'isabelline', 'italian sky blue',
              'ivory', 'jade', 'japanese carmine', 'japanese violet', 'jasmine', 'jazzberry jam', 'jet', 'jonquil',
              'june bud', 'jungle green', 'kelly green', 'keppel', 'key lime', 'Khaki (web) ', 'Khaki', 'Khaki (X11) ',
              'Light khaki', 'kobe', 'kobi', 'kobicha', 'kombu green', 'ksu purple', 'la salle green',
              'languid lavender', 'lanzones', 'lapis lazuli', 'laser lemon', 'laurel green', 'lava', 'Lavender ',
              'floral', 'Lavender ', 'web', 'lavender blue', 'lavender blush', 'lavender gray', 'lavender indigo',
              'lavender magenta', 'lavender mist', 'lavender pink', 'lavender purple', 'lavender rose', 'lawn green',
              'lemon', 'lemon chiffon', 'lemon curry', 'lemon glacier', 'lemon iced tea', 'lemon lime', 'lemon lime',
              'lemon meringue', 'lemon yellow', 'Lemon yellow ', 'Crayola', 'lenurple', 'liberty', 'licorice',
              'light apricot', 'light blue', 'light brown', 'light carmine pink', 'light chocolate cosmos',
              'light cobalt blue', 'light coral', 'light cornflower blue', 'light crimson', 'light cyan',
              'light deep pink', 'light french beige', 'light fuchsia pink', 'light gold', 'light goldenrod',
              'light goldenrod yellow', 'light gray', 'light grayish magenta', 'light green', 'light hot pink',
              'light khaki', 'light medium orchid', 'light moss green', 'light mustard', 'light orange', 'light orchid',
              'light pastel purple', 'light periwinkle', 'light pink', 'light red', 'light red ochre', 'light salmon',
              'light salmon pink', 'light sea green', 'light silver', 'light sky blue', 'light slate gray',
              'light steel blue', 'light taupe', 'light thulian pink', 'light turquoise', 'light violet',
              'light yellow', 'lilac', 'lilac luster', 'Lime ', 'color wheel', 'Lime (web) ', 'X11 green', 'lime green',
              'limerick', 'lincoln green', 'linen', 'lion', 'liseran purple', 'little boy blue', 'little girl pink',
              'liver', 'Liver ', 'dogs', 'Liver ', 'organ', 'liver chestnut', 'livid', 'lotion', 'lotion blue',
              'lotion pink', 'lumber', 'lust', 'maastricht blue', 'macaroni and cheese', 'madder lake', 'magenta',
              'Magenta ', 'Pantone', 'mahogany', 'maize', 'Maize ', 'Crayola', 'majorelle blue', 'malachite', 'manatee',
              'mandarin', 'mango', 'mango green', 'mango tango', 'mantis', 'mardi gras', 'marigold', 'Maroon ',
              'Crayola', 'Maroon ', 'HTML/CSS', 'Maroon ', 'X11', 'mauve', 'mauve taupe', 'mauvelous', 'maximum blue',
              'maximum blue green', 'maximum blue purple', 'maximum green', 'maximum green yellow', 'maximum orange',
              'maximum purple', 'maximum pink', 'maximum red', 'maximum red purple', 'maximum violet', 'maximum yellow',
              'maximum yellow red', 'may green', 'maya blue', 'meat brown', 'medium aquamarine', 'medium blue',
              'medium candy apple red', 'medium carmine', 'medium champagne', 'medium electric blue', 'medium green',
              'medium jungle green', 'medium lavender magenta', 'medium orange', 'medium orchid', 'medium persian blue',
              'medium pink', 'medium purple', 'medium red', 'medium red-violet', 'medium ruby', 'medium sea green',
              'medium sky blue', 'medium slate blue', 'medium spring bud', 'medium spring green', 'medium taupe',
              'medium turquoise', 'medium tuscan red', 'medium vermilion', 'medium violet', 'medium violet-red',
              'medium yellow', 'mellow apricot', 'mellow yellow', 'melon', 'Melon ', 'Crayola', 'menthol',
              'metallic blue', 'metallic bronze', 'metallic brown', 'metallic gold', 'metallic green',
              'metallic orange', 'metallic pink', 'metallic red', 'metallic seaweed', 'metallic silver',
              'metallic sunburst', 'metallic violet', 'metallic yellow', 'mexican pink', 'microsoft blue',
              'microsoft edge blue', 'microsoft green', 'microsoft red', 'microsoft yellow', 'middle blue',
              'middle blue green', 'middle blue purple', 'middle grey', 'middle green', 'middle green yellow',
              'middle purple', 'middle red', 'middle red purple', 'middle yellow', 'middle yellow red', 'midnight',
              'midnight blue', 'midnight blue', 'Midnight green ', 'eagle green', 'mikado yellow', 'milk',
              'milk chocolate', 'mimi pink', 'mindaro', 'ming', 'minion yellow', 'mint', 'mint cream', 'mint green',
              'misty moss', 'misty rose', 'moccasin', 'mocha', 'mode beige', 'moonstone', 'moonstone blue',
              'mordant red 19', 'morning blue', 'moss green', 'mountain meadow', 'mountbatten pink', 'msu green', 'mud',
              'mughal green', 'mulberry', 'Mulberry ', 'Crayola', "mummy's tomb", 'mustard', 'mustard brown',
              'mustard green', 'mustard yellow', 'myrtle green', 'mystic', 'mystic maroon', 'mystic red',
              'nadeshiko pink', 'napier green', 'naples yellow', 'navajo white', 'navy blue', 'Navy blue ', 'Crayola',
              'navy purple', 'neon blue', 'neon brown', 'neon carrot', 'neon cyan', 'neon fuchsia', 'neon gold',
              'neon gray', 'neon dark green', 'neon green', 'neon green', 'neon pink', 'neon purple', 'neon red',
              'neon scarlet', 'neon silver', 'neon tangerine', 'neon yellow', 'new car', 'new york pink', 'nickel',
              'nintendo red', 'non-photo blue', 'nyanza', 'ocean blue', 'ocean boat blue', 'ocean green', 'ochre',
              'office green', 'ogre odor', 'old burgundy', 'old gold', 'old heliotrope', 'old lace', 'old lavender',
              'old mauve', 'old moss green', 'old rose', 'old silver', 'olive', 'Olive drab ', '#3', 'olive drab #7',
              'olive green', 'olivine', 'onyx', 'opal', 'opera mauve', 'orange', 'Orange ', 'color wheel', 'Orange ',
              'Crayola', 'Orange ', 'Pantone', 'Orange ', 'RYB', 'Orange ', 'web', 'orange iced tea', 'orange peel',
              'orange-red', 'Orange-red ', 'Crayola', 'orange soda', 'orange soda', 'orange-yellow', 'Orange-yellow ',
              'Crayola', 'orchid', 'orchid pink', 'Orchid ', 'Crayola', 'orioles orange', 'otter brown', 'outer space',
              'Outer space ', 'Crayola', 'outrageous orange', 'oxblood', 'oxford blue', 'oxley', 'ou crimson red',
              'pacific blue', 'pakistan green', 'palatinate blue', 'palatinate purple', 'pale aqua', 'pale blue',
              'pale brown', 'pale carmine', 'pale cerulean', 'pale chestnut', 'pale copper', 'pale cornflower blue',
              'pale cyan', 'pale gold', 'pale goldenrod', 'pale green', 'pale lavender', 'pale magenta',
              'pale magenta-pink', 'pale pink', 'pale plum', 'pale red-violet', 'pale robin egg blue', 'pale silver',
              'pale spring bud', 'pale taupe', 'pale turquoise', 'pale violet', 'pale violet-red', 'palm leaf',
              'pansy purple', 'paolo veronese green', 'papaya whip', 'paradise pink', 'parchment', 'paris green',
              'parrot pink', 'pastel blue', 'pastel brown', 'pastel gray', 'pastel green', 'pastel magenta',
              'pastel orange', 'pastel pink', 'pastel purple', 'pastel red', 'pastel violet', 'pastel yellow',
              'patriarch', "payne's grey", 'peach', 'Peach ', 'Crayola', 'peach-orange', 'peach puff', 'peach-yellow',
              'pear', 'pearl', 'pearl aqua', 'pearly purple', 'peridot', 'periwinkle', 'Periwinkle ', 'Crayola',
              'permanent geranium lake', 'persian blue', 'persian green', 'persian indigo', 'persian orange',
              'persian pink', 'persian plum', 'persian red', 'persian rose', 'persimmon', 'peru', 'petal',
              'pewter blue', 'philippine blue', 'philippine bronze', 'philippine brown', 'philippine gold',
              'philippine golden yellow', 'philippine gray', 'philippine green', 'philippine indigo',
              'philippine orange', 'philippine pink', 'philippine red', 'philippine silver', 'philippine sky blue',
              'philippine violet', 'philippine yellow', 'phlox', 'phthalo blue', 'phthalo green', 'picton blue',
              'pictorial carmine', 'piggy pink', 'pine green', 'pine tree', 'pineapple', 'pink', 'Pink ', 'Pantone',
              'Pink Diamond ', 'Ace Hardware Color', 'Pink Diamond ', 'Independent Retailers Colors', 'pink flamingo',
              'pink lace', 'pink lavender', 'pink-orange', 'pink pearl', 'pink raspberry', 'pink sherbet', 'pistachio',
              'pixie powder', 'platinum', 'plum', 'Plum ', 'web', 'plump purple', 'poison purple', 'police blue',
              'polished pine', 'pomp and power', 'popstar', 'portland orange', 'powder blue', 'prilly blue',
              'prilly pink', 'prilly red', 'princess perfume', 'princeton orange', 'prune', 'prussian blue',
              'psychedelic purple', 'puce', 'puce red', 'Pullman Brown ', 'UPS Brown', 'pullman green', 'pumpkin',
              'Purple ', 'HTML', 'Purple ', 'Munsell', 'Purple ', 'X11', 'purple heart', 'purple mountain majesty',
              'purple navy', 'purple pizzazz', 'purple plum', 'purple taupe', 'purpureus', 'quartz', 'queen blue',
              'queen pink', 'quick silver', 'quinacridone magenta', 'quincy', 'rackley', 'radical red', 'raisin black',
              'rajah', 'raspberry', 'raspberry glace', 'raspberry pink', 'raspberry rose', 'raw sienna', 'raw umber',
              'razzle dazzle rose', 'razzmatazz', 'razzmic berry', 'rebecca purple', 'red', 'Red ', 'Crayola', 'Red ',
              'Munsell', 'Red ', 'NCS', 'Red ', 'Pantone', 'Red ', 'pigment', 'Red ', 'RYB', 'red-brown', 'red cola',
              'red devil', 'red-orange', 'Red-orange ', 'Crayola', 'Red-orange ', 'Color wheel', 'red-purple',
              'red rum', 'red salsa', 'red strawberry', 'red-violet', 'Red-violet ', 'Crayola', 'Red-violet ',
              'Color wheel', 'redwood', 'registration black', 'resolution blue', 'rhythm', 'rich brilliant lavender',
              'rich carmine', 'rich electric blue', 'rich lavender', 'rich lilac', 'rich maroon', 'rifle green',
              'ripe mango', 'roast coffee', 'robin egg blue', 'rocket metallic', 'roman silver', 'root beer', 'rose',
              'rose bonbon', 'rose dust', 'rose ebony', 'rose garnet', 'rose gold', 'rose madder', 'rose pink',
              'rose quartz', 'rose quartz pink', 'rose red', 'rose taupe', 'rose vale', 'rosewood', 'rosy brown',
              'royal azure', 'royal blue', 'royal blue', 'royal brown', 'royal fuchsia', 'royal green', 'royal orange',
              'royal pink', 'royal red', 'royal red', 'royal purple', 'royal yellow', 'ruber', 'rubine red', 'ruby',
              'ruby red', 'rufous', 'rum', 'russet', 'russian green', 'russian violet', 'rust', 'rusty red',
              'sacramento state green', 'saddle brown', 'safety orange', 'Safety orange ', 'blaze orange',
              'safety yellow', 'saffron', 'sage', "st. patrick's blue", 'salem', 'salmon', 'salmon rose', 'salmon pink',
              'samsung blue', 'sand', 'sand dune', 'sandstorm', 'sandy brown', 'sandy tan', 'sandy taupe', 'sap green',
              'sapphire', 'sapphire blue', 'sasquatch socks', 'satin sheen gold', 'scarlet', 'Scarlet ', 'Crayola',
              'schauss pink', 'school bus yellow', "screamin' green", 'sea blue', 'sea foam green', 'sea green',
              'Sea green ', 'Crayola', 'sea serpent', 'seal brown', 'seashell', 'selective yellow', 'sepia', 'shadow',
              'shadow blue', 'shampoo', 'shamrock green', 'shandy', 'sheen green', 'shimmering blush', 'shiny shamrock',
              'shocking pink', 'Shocking pink ', 'Crayola', 'sienna', 'silver', 'Silver ', 'Crayola', 'Silver ',
              'Metallic', 'silver chalice', 'silver foil', 'silver lake blue', 'silver pink', 'silver sand', 'sinopia',
              'sizzling red', 'sizzling sunrise', 'skobeloff', 'sky blue', 'Sky blue ', 'Crayola', 'sky magenta',
              'slate blue', 'slate gray', 'slimy green', 'Smalt ', 'Dark powder blue', 'smashed pumpkin', 'smitten',
              'smoke', 'smokey topaz', 'smoky black', 'snow', 'soap', 'solid pink', 'sonic silver', 'spartan crimson',
              'space cadet', 'spanish bistre', 'spanish blue', 'spanish carmine', 'spanish crimson', 'spanish gray',
              'spanish green', 'spanish orange', 'spanish pink', 'spanish purple', 'spanish red', 'spanish sky blue',
              'spanish violet', 'spanish viridian', 'spanish yellow', 'spicy mix', 'spiro disco ball', 'spring bud',
              'spring frost', 'spring green', 'Spring green ', 'Crayola', 'star command blue', 'steel blue',
              'steel pink', 'steel teal', 'stil de grain yellow', 'straw', 'strawberry', 'stop red',
              'strawberry iced tea', 'strawberry red', 'sugar plum', 'sunburnt cyclops', 'sunglow', 'sunny', 'sunray',
              'sunset', 'sunset orange', 'super pink', 'sweet brown', 'taffy', 'tan', 'Tan ', 'Crayola', 'tangelo',
              'tangerine', 'tangerine yellow', 'tango pink', 'tart orange', 'taupe', 'taupe gray', 'tea green',
              'tea rose', 'tea rose', 'teal', 'teal blue', 'teal deer', 'teal green', 'telemagenta', 'temptress',
              'Tenné ', 'tawny', 'terra cotta', 'thistle', 'Thistle ', 'Crayola', 'thulian pink', 'tickle me pink',
              'tiffany blue', "tiger's eye", 'timberwolf', 'titanium', 'titanium yellow', 'tomato', 'tomato sauce',
              'toolbox', 'tooth', 'topaz', 'tractor red', 'trolley grey', 'tropical rain forest', 'tropical violet',
              'true blue', 'tufts blue', 'tulip', 'tumbleweed', 'turkish rose', 'turquoise', 'turquoise blue',
              'turquoise green', 'turquoise surf', 'turtle green', 'tuscan', 'tuscan brown', 'tuscan red', 'tuscan tan',
              'tuscany', 'twilight lavender', 'twitter blue', 'tyrian purple', 'ube', 'ultramarine', 'ultramarine blue',
              'Ultramarine blue ', "Caran d'Ache", 'ultra pink', 'ultra red', 'umber', 'unbleached silk',
              'united nations blue', 'unmellow yellow', 'up maroon', 'upsdell red', 'urobilin', 'vampire black',
              'van dyke brown', 'vanilla', 'vanilla ice', 'vegas gold', 'venetian red', 'verdigris', 'vermilion',
              'vermilion', 'veronica', 'verse green', 'very light azure', 'very light blue',
              'very light malachite green', 'very light tangelo', 'very pale orange', 'very pale yellow', 'vine green',
              'violet', 'Violet ', "Caran d'Ache", 'Violet ', 'color wheel', 'Violet ', 'crayola', 'Violet ', 'RYB',
              'Violet ', 'web', 'violet-blue', 'Violet-blue ', 'Crayola', 'violet-red', 'violin brown', 'viridian',
              'viridian green', 'vista blue', 'vivid amber', 'vivid auburn', 'vivid burgundy', 'vivid cerise',
              'vivid cerulean', 'vivid crimson', 'vivid gamboge', 'vivid lime green', 'vivid malachite',
              'vivid mulberry', 'vivid orange', 'vivid orange peel', 'vivid orchid', 'vivid raspberry', 'vivid red',
              'vivid red-tangelo', 'vivid sky blue', 'vivid tangelo', 'vivid tangerine', 'vivid vermilion',
              'vivid violet', 'vivid yellow', 'water', 'watermelon', 'watermelon red', 'watermelon yellow',
              'waterspout', 'weldon blue', 'wenge', 'wheat', 'white', 'white chocolate', 'white coffee', 'white smoke',
              'wild orchid', 'wild strawberry', 'wild watermelon', 'willpower orange', 'windsor tan', 'wine',
              'wine dregs', 'wine red', 'winter sky', 'winter wizard', 'wintergreen dream', 'wisteria', 'wood brown',
              'xanadu', 'yellow', 'Yellow ', 'Crayola', 'Yellow ', 'Munsell', 'Yellow ', 'NCS', 'Yellow ', 'Pantone',
              'Yellow ', 'process', 'Yellow ', 'RYB', 'yellow-green', 'Yellow-green ', 'Crayola', 'yellow orange',
              'Yellow Orange ', 'Color Wheel', 'yellow rose', 'yellow sunshine', 'yinmn blue', 'zaffre', 'zebra white',
              'zinnwaldite', 'zomp']

    # binary: yes/no
    # numbers
    def is_number(s):
        if s.isnumeric() or s.isdigit():
            return True
        try:
            float(s)
            return True
        except ValueError:
            return False

    answer_vec = torch.zeros(len(answer_to_index))
    for answer in answers:
        if answer in ['yes', 'no']:
            answer = "binary"
        elif answer in ["unanswerable", 'unsuitable', 'unsuitable image']:
            answer = "unanswerable"
        elif is_number(answer):
            answer = "number"
        elif answer in colors:
            answer = "color"
        else:
            answer = "other"

        index = answer_to_index.get(answer)
        if index is not None:
            answer_vec[index] += 1
    return answer_vec


