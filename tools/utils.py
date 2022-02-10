import random
import string


def annotations_list(annotations):
    """Given an annotation file, return a list of them."""
    with open(annotations) as ann:
        result = [line.split(' ')[1].strip().replace('-', '_') for line in ann]
    return result


def annotations_dic(annotations):
    """Given an annotation file, return a dictionary {label: index} of them."""
    labels = annotations_list(annotations)
    return {label: i for i, label in enumerate(labels)}


def gen_id(size=8):
    """Generate a random id."""
    chars = string.ascii_uppercase + string.digits
    return ''.join(random.choice(chars) for _ in range(size))
