import random
import string


def annotations_list(annotations):
    """Given an annotation file, return a list of them."""
    with open(annotations) as ann:
        result = [line.strip().replace('-', '_') for line in ann]
    return result


def annotations_dic(annotations):
    """Given an annotation file, return a dictionary {label: index} of them."""
    labels = annotations_list(annotations)
    return {label: i for i, label in enumerate(labels)}


def annotations_dict_rev(annotations):
    """Given an annotation file return a dictionary {index: label} of them."""
    result = annotations_dic(annotations)
    return {v: k for k, v in result.items()}


def gen_id(size=8):
    """Generate a random id."""
    chars = string.ascii_uppercase + string.digits
    return ''.join(random.choice(chars) for _ in range(size))


def prettify(byte_content):
    """Prettify subprocess output.

    Args:
        byte_content ([type]): [description]

    Returns:
        [type]: [description]
    """
    decoded = byte_content.decode('utf-8')
    formatted_output = decoded.replace('\\n', '\n').replace('\\t', '\t')
    return formatted_output
