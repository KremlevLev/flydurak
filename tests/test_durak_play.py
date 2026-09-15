from flysans.durak_play import action_name,card_name
from flysans.durak import PASS,TAKE

def test_human_card_labels():
    assert card_name(0)=="6♣";assert card_name(35)=="A♠"
    assert action_name(TAKE)=="БЕРУ";assert action_name(PASS)=="ПАС"
