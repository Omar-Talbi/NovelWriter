import re


def test_hero_age_constant():
    with open('rag/story_bible.json', 'r') as f:
        text = f.read()
    assert re.search(r'Hero is 17', text), 'Hero age must remain 17'
