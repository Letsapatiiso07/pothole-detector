from collections import Counter

from PIL import Image

from data import crop_road, group_split, parse_annotations, select_sample, session_of


def test_group_split_keeps_each_group_in_one_split():
    groups = ['a'] * 50 + ['b'] * 30 + ['c'] * 10 + ['d'] * 8 + ['e'] * 2
    splits = group_split(groups, test_frac=0.2, val_frac=0.1, seed=1)
    assert len(splits) == len(groups)
    by_group = {}
    for g, s in zip(groups, splits):
        by_group.setdefault(g, set()).add(s)
    assert all(len(s) == 1 for s in by_group.values())
    sizes = Counter(splits)
    assert sizes['train'] > 0 and sizes['test'] > 0
    assert sizes['test'] <= 1.2 * 0.2 * len(groups)


def test_group_split_is_repeatable():
    groups = [str(i % 7) for i in range(200)]
    assert group_split(groups, seed=3) == group_split(groups, seed=3)


def test_select_sample_balances_sessions_and_drops_clashes():
    pos = ['G0010001.JPG', 'G0010002.JPG', 'G0010003.JPG', 'G0020001.JPG', 'G0030001.JPG']
    neg = ['G0010009.JPG', 'G0010010.JPG', 'G0020001.JPG', 'G0020005.JPG', 'G0020006.JPG']
    chosen, clash = select_sample(pos, neg, per_class=100)
    assert clash == {'G0020001.JPG'}
    names = [n for n, _ in chosen]
    assert 'G0020001.JPG' not in names
    count = Counter((session_of(n), label) for n, label in chosen)
    # session 001: 3 pothole, 2 normal -> 2 each; 002: 0 pothole left; 003: no normal
    assert count == {('001', 'pothole'): 2, ('001', 'normal'): 2}


def test_parse_annotations_reads_boxes_and_drops_repeats():
    text = ('Train data\\Positive data\\G0010033.bmp 2 10 20 30 40 50 60 70 80   \n'
            'Train data\\Positive data\\G0010033.bmp 2 10 20 30 40 50 60 70 80\n'
            'Train data\\Positive data\\G0010034.bmp 1 1 2 3 4\n')
    boxes = parse_annotations(text)
    assert boxes == {'G0010033.JPG': [(10, 20, 30, 40), (50, 60, 70, 80)],
                     'G0010034.JPG': [(1, 2, 3, 4)]}


def test_crop_road_keeps_full_width():
    strip = crop_road(Image.new('RGB', (920, 690)))
    assert strip.size == (920, 200)
