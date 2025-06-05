from pathlib import Path
from rag.story_bible import StoryBible, Fact
import time


def test_story_bible_search(tmp_path):
    path = tmp_path / 'bible.json'
    bible = StoryBible(path)
    bible.add_fact(Fact(id='1', type='character', text='Hero is 17 years old', timestamp=time.time()))
    bible.add_fact(Fact(id='2', type='world', text='The city is called Avalor', timestamp=time.time()))
    results = bible.search('hero age', k=1)
    assert results and '17' in results[0].text
