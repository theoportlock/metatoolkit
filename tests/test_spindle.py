import pytest
from metatoolkit.spindle import *
import pandas as pd

def test_placeholder():
    # Placeholder test
    assert True

def test_parse_arguments():
    args = ['test_subject', '--meta', 'metadata.tsv']
    parsed_args = parse_arguments(args)
    assert parsed_args.subject == 'test_subject'
    assert parsed_args.meta == 'metadata.tsv'
