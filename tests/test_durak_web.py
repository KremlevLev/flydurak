from flysans.durak import DurakGame
from flysans.durak_web import activity_svg, card_html, hand_html, sort_hand

def test_web_cards_are_large_click_targets():
    rendered=card_html(35,True)
    assert 'button' in rendered and 'A' in rendered and '♠' in rendered

def test_hand_groups_suits_and_puts_sorted_trumps_last():
    cards = [35, 27, 2, 0, 20, 18, 11, 9]
    assert sort_hand(cards, 0) == [9, 11, 18, 20, 27, 35, 0, 2]
    rendered = hand_html(cards, set(cards), 0)
    assert rendered.rfind("КОЗЫРИ") > rendered.rfind("♠")

def test_activity_svg_shows_neuron_ids_and_both_signs():
    rendered = activity_svg([(12345, .9), (67890, -.4)], coordinates={12345: (20, 20), 67890: (40, 40)}, anatomy=[(20, 20), (40, 40)])
    assert "neuron 12345" in rendered
    assert "#ffe5a3" in rendered and "#b8f1ff" in rendered
