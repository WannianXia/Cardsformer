import numpy as np
import clr
import random
import os
import System
from System.Collections import *
# sys.path.append("/home/xingdp/xia_wannian/HearthStoneAI/SabberStone-master/core-extensions/SabberStoneBasicAI/bin/Debug/netstandard2.0")
clr.AddReference(
    os.getcwd() + "/Env/DllSimulator/SabberStoneCore.dll")
clr.AddReference(
    os.getcwd() + "/Env/DllSimulator/SabberStoneBasicAI.dll")
import SabberStoneCore
import SabberStoneBasicAI
from SabberStoneBasicAI.Score import *
from SabberStoneBasicAI.Nodes import *
from SabberStoneBasicAI.Meta import *
from SabberStoneCore.Enums import *
from SabberStoneCore.Config import *
from SabberStoneCore.Model import *
from SabberStoneCore.Tasks.PlayerTasks import ChooseTask

from Env.Deck import Deck

def check_race(card_entity):
    """
        Get the minion race
    """
    race_list = [0, 18, 15, 20, 14, 21, 23, 17, 24]
    race_id = None
    for (i, id) in enumerate(race_list):
        if card_entity.IsRace(id):
            race_id = i
            break
    return race_id

def check_type(card_type):
    """
        Get the card type
    """
    type_list = [4, 5, 7]
    return type_list.index(card_type)
    # [0: minion, 1: spell, 2: weapon]

def DeckList(deck_list, random_cards=False):
    deck = System.Collections.Generic.List[Card]()
    for card_name in deck_list:
        card = Cards.FromName(card_name)
        if random_cards == True:
            # To decouple the description and corresponding scalar vector, randomize the Attack, Health, Cost of a card.
            if card.Type == 4:
                card.ATK = random.randint(0, 12)
                card.Health = random.randint(1, 12)
                card.Cost = random.randint(0, 10)
            elif card.Type == 5:
                card.Cost = random.randint(0, 10)
            elif card.Type == 7:
                card.Cost = random.randint(0, 10)
                card.ATK = random.randint(0, 10)
        if card is None:
            raise Exception("Card Is None Exception {}".format(card_name))
        else:
            deck.Add(Cards.FromName(card_name))

    return deck

def get_id(name):
    return card_name.index(name)
        
def modify_cards():
    # Some cards has been updated by the official game, while SabberStone has not yet implemented
    soulfire = Cards.FromName("Soulfire")
    soulfire.Cost = 0

    knifeujuggler = Cards.FromName("Knife Juggler")
    knifeujuggler.ATK = 3

    leeroy = Cards.FromName("Leeroy Jenkins")
    leeroy.Cost = 4

    hunter_mark = Cards.FromName("Hunter's Mark")
    hunter_mark.Cost = 0

    flare = Cards.FromName("Flare")
    flare.Cost = 1

    starving_buzzard = Cards.FromName("Starving Buzzard")
    starving_buzzard.Cost = 2
    starving_buzzard.ATK = 2
    starving_buzzard.Health = 1

    mana_wyrm = Cards.FromName("Mana Wyrm")
    mana_wyrm.Cost = 1

    equality = Cards.FromName("Equality")
    equality.Cost = 2

    return


class Hearthstone:
    def __init__(self, random_cards=False, player1_name="Player1", player2_name="Player2", start_player=1,
                 fill_decks=False, shuffle=True, skip_mulligan=False, logging=False, history=False,
                 player1_deck=None, player2_deck=None):
        game_config = GameConfig()
        game_config.StartPlayer = start_player
        game_config.Player1Name = player1_name
        game_config.Player2Name = player2_name
        game_config.FillDecks = fill_decks
        game_config.Shuffle = shuffle
        game_config.SkipMulligan = skip_mulligan
        game_config.Logging = logging
        game_config.History = history
        self.game_config = game_config
        modify_cards()
        self.player1_deck = player1_deck
        self.player2_deck = player2_deck
        self.random_cards = random_cards


    def step(self, action):
        self.game.Process(action)
        position = self.game.CurrentPlayer.Name
        obs, options = self.get_model_input(self.get_current_state())
        reward = 0
        done = self.game.State == State.COMPLETE

        if done:
            won = self.game.CurrentPlayer.PlayState == PlayState.WON
            cur_player = 1 if self.game.CurrentPlayer.Name == "Player1" else 2
            if won and cur_player == 1:
                reward += 1
            elif won and cur_player == 2:
                reward -= 1
            elif not won and cur_player == 1:
                reward -= 1
            elif not won and cur_player == 2:
                reward += 1
            else:
                raise RuntimeError("The reward is not defined")
        return position, obs, options, reward, done

    def reset(self):
        p1_deck = Deck.deck_list[random.randint(0, len(Deck.deck_list)-1)] if self.player1_deck is None else self.player1_deck
        p2_deck = Deck.deck_list[random.randint(0, len(Deck.deck_list)-1)] if self.player2_deck is None else self.player2_deck

        self.game_config.Player1HeroClass = p1_deck["Class"]
        self.game_config.Player1Deck = DeckList(p1_deck["Deck"], self.random_cards)
        self.game_config.Player2HeroClass = p2_deck["Class"]
        self.game_config.Player2Deck = DeckList(p2_deck["Deck"], self.random_cards)
        self.game = Game(self.game_config)
        self.game.StartGame()
        self.game.Process(ChooseTask.Mulligan(
            self.game.Player1, System.Collections.Generic.List[int]()))
        self.game.Process(ChooseTask.Mulligan(
            self.game.Player2, System.Collections.Generic.List[int]()))
        self.game.MainReady()
        position = self.game.CurrentPlayer.Name
        state = self.get_current_state()
        obs, options = self.get_model_input(state)
        done = self.game.State == State.COMPLETE
        reward = 0
        return position, obs, options, reward, done
    

    def player_state(self, current = True):
        """
            Get the scalar feature vector of a player entity
        """
        hero_state = np.zeros((2, 31))
        if current:
            player_list = [self.game.CurrentPlayer, self.game.CurrentOpponent]
        else:
            player_list = [self.game.CurrentOpponent, self.game.CurrentPlayer]
        for i in range(2):
            entity = player_list[i]
            hero_state[i, 0] = 1 if entity.Hero.CanAttack else 0
            hero_state[i, 1] = entity.Hero.AttackDamage
            hero_state[i, 2] = entity.Hero.BaseHealth
            hero_state[i, 3] = entity.Hero.Health
            hero_state[i, 4] = 1 if entity.Hero.IsFrozen else 0
            hero_state[i, 5] = 1 if entity.Hero.HasWindfury else 0
            hero_state[i, 6] = 1 if entity.Hero.HasStealth else 0
            hero_state[i, 7] = entity.RemainingMana
            hero_state[i, 8] = entity.BaseMana
            hero_state[i, 9] = entity.CurrentSpellPower
            hero_state[i, 10] = entity.Hero.Armor
            if entity.Hero.Weapon is not None:
                hero_state[i, 11] = 1
                hero_state[i, 12] = entity.Hero.Weapon.Durability
                hero_state[i, 13] = entity.Hero.Weapon.Card.ATK
            hero_state[i, 14] = entity.DeckZone.Count
            hero_state[i, 15] = entity.HandZone.Count
            hero_state[i, 16] = entity.SecretZone.Count
            hero_state[i, 17 + entity.BaseClass - 2] = 1
            if i < 1:
                hero_state[i, 26] = entity.Hero.NumAttacksThisTurn
                hero_state[i, 27] = entity.NumCardsPlayedThisTurn
                hero_state[i, 28] = entity.OverloadLocked

        return hero_state



    def hand_card_state(self, current = True):

        """
            Get the scalar feature vector of a hand card entity
        """
        card_feat = np.zeros((11, 23))
        if current:
            handzone = self.game.CurrentPlayer.HandZone
        else:
            handzone = self.game.CurrentOpponent.HandZone
        for (i, entity) in enumerate(handzone):
            card_feat[i, 0] = entity.Cost
            card_feat[i, 1] = entity.Card.Cost
            card_feat[i, 2] = 1 if entity.IsPlayable else 0
            # [0: minion, 1: spell, 2: weapon]
            type_id = check_type(entity.Card.Type)
            if type_id == 0:  # minion
                card_feat[i, 3] = entity.Card.ATK
                card_feat[i, 4] = entity.AttackDamage
                card_feat[i, 5] = entity.Card.Health
                card_feat[i, 6] = entity.BaseHealth
                card_feat[i, 7 + check_race(entity)] = 1
            elif type_id == 2:  # weapon
                card_feat[i, 3] = entity.Card.ATK
                card_feat[i, 4] = entity.AttackDamage
                card_feat[i, 5] = entity.Durability
                card_feat[i, 6] = entity.Durability
            card_feat[i, 16 + type_id] = 1
        return card_feat


    def board_minion_state(self, current = True):
        
        """
            Get the scalar feature vector of a minion entity
        """
        minions = np.zeros((14, 26))
        if current:
            board_zone_list = [self.game.CurrentPlayer.BoardZone, self.game.CurrentOpponent.BoardZone]
        else:
            board_zone_list = [self.game.CurrentOpponent.BoardZone, self.game.CurrentPlayer.BoardZone]
        for (i, entity) in enumerate(board_zone_list[0]):
            # minions[i, 0] = entity.Cost
            minions[i, 0] = entity.Card.Cost
            minions[i, 1] = 1 if entity.CanAttack else 0
            minions[i, 2] = entity.Card.ATK
            minions[i, 3] = entity.AttackDamage
            minions[i, 4] = entity.Health
            minions[i, 5] = entity.BaseHealth
            minions[i, 6] = 1 if entity.HasTaunt else 0
            minions[i, 7] = 1 if entity.HasDivineShield else 0
            minions[i, 8] = 1 if entity.HasDeathrattle else 0
            minions[i, 9] = 1 if entity.IsFrozen else 0
            minions[i, 10] = 1 if entity.HasWindfury else 0
            minions[i, 11] = 1 if entity.IsSilenced else 0
            minions[i, 12] = 1 if entity.HasStealth else 0
            minions[i, 13] = entity.NumAttacksThisTurn
            minions[i, 14 + check_race(entity)] = 1
        for (j, entity) in enumerate(board_zone_list[1]):
            i = j + 7
            minions[i, 0] = entity.Card.Cost
            minions[i, 1] = 1 if entity.CanAttack else 0
            minions[i, 2] = entity.Card.ATK
            minions[i, 3] = entity.AttackDamage
            minions[i, 4] = entity.Health
            minions[i, 5] = entity.BaseHealth
            minions[i, 6] = 1 if entity.HasTaunt else 0
            minions[i, 7] = 1 if entity.HasDivineShield else 0
            minions[i, 8] = 1 if entity.HasDeathrattle else 0
            minions[i, 9] = 1 if entity.IsFrozen else 0
            minions[i, 10] = 1 if entity.HasWindfury else 0
            minions[i, 11] = 1 if entity.IsSilenced else 0
            minions[i, 12] = 1 if entity.HasStealth else 0
            minions[i, 13] = entity.NumAttacksThisTurn
            minions[i, 14 + check_race(entity)] = 1

        return minions

    def get_current_state(self, current = True):
        if current:
            current_player = self.game.CurrentPlayer
            opponent_player = self.game.CurrentOpponent
        else:
            current_player = self.game.CurrentOpponent
            opponent_player = self.game.CurrentPlayer
        hand_card_names = [None, ] * 11
        minion_names = [None, ] * 14
        weapon_names = [
            current_player.Hero.Weapon.Card.Name if current_player.Hero.Weapon is not None else None, 
            opponent_player.Hero.Weapon.Card.Name if opponent_player.Hero.Weapon is not None else None
            ]
        secret_names = [None, ] * 5

        hand_card_scalar = self.hand_card_state(current)

        minion_scalar = self.board_minion_state(current)

        hero_scalar = self.player_state(current)


        for (i, hand_card) in enumerate(current_player.HandZone):
            hand_card_names[i] = hand_card.Card.Name

        # check hero power state, add as a hand card if available
        hand_num = current_player.HandZone.Count
        if not current_player.Hero.HeroPower.IsExhausted:
            hand_card_names[hand_num] = current_player.Hero.HeroPower.Card.Name
            hand_card_scalar[hand_num, 0] = current_player.Hero.HeroPower.Cost
            hand_card_scalar[hand_num, 1] = current_player.Hero.HeroPower.Card.Cost
            hand_card_scalar[hand_num, 2] = 1 if current_player.Hero.HeroPower.IsPlayable else 0
        for (i, board_minion) in enumerate(current_player.BoardZone):
            minion_names[i] = board_minion.Card.Name
        for (i, board_minion) in enumerate(opponent_player.BoardZone):
            minion_names[i + 7] = board_minion.Card.Name
        for (i, secret) in enumerate(current_player.SecretZone):
            secret_names[i] = secret.Card.Name

        cur_state = {
            "hand_card_names": hand_card_names,
            "minion_names": minion_names,
            "weapon_names": weapon_names,
            "secret_names": secret_names,
            "hand_card_scalar": hand_card_scalar,
            "minion_scalar": minion_scalar,
            "hero_scalar": hero_scalar,
        }

        return cur_state


    def get_model_input(self, game_state):
        options = self.game.CurrentPlayer.Options()
        num_options = len(options)
        hand_card_scalar_batch = np.repeat(
            game_state["hand_card_scalar"][np.newaxis, :], num_options, axis=0)  # [num, 11, 5]
        minion_scalar_batch = np.repeat(game_state["minion_scalar"][np.newaxis, :], num_options, axis=0)
        hero_scalar_batch = np.repeat(game_state["hero_scalar"][np.newaxis, :], num_options, axis=0)
        hand_num = self.game.CurrentPlayer.HandZone.Count
        for i in range(num_options):
            option = options[i]
            option_name = type(option).__name__
            if option_name == 'EndTurnTask':
                continue
            elif option_name == 'HeroPowerTask':
                hand_card_scalar_batch[i, hand_num, -1] = 1
                if option.HasTarget:
                    if option.Target.Zone is not None:
                        if option.Target.Zone.Controller.Name == self.game.CurrentPlayer.Name:
                            minion_scalar_batch[i, option.Target.ZonePosition, -2] = 1
                        elif option.Target.Zone.Controller.Name == self.game.CurrentOpponent.Name:
                            minion_scalar_batch[i, 7 + option.Target.ZonePosition, -2] = 1
                    elif option.Target == self.game.CurrentPlayer.Hero:
                        hero_scalar_batch[i, 0, -2] = 1
                    elif option.Target == self.game.CurrentOpponent.Hero:
                        hero_scalar_batch[i, 1, -2] = 1
            elif option_name == 'PlayCardTask':
                hand_card_scalar_batch[i, option.Source.ZonePosition, -1] = 1
                if option.ZonePosition != -1:
                    # play minions, the target has a 'position' flag in -3
                    minion_scalar_batch[i, option.ZonePosition, -3] = 1
                if option.HasTarget:
                    if option.Target.Zone is not None:
                        if option.Target.Zone.Controller.Name == self.game.CurrentPlayer.Name:
                            minion_scalar_batch[i, option.Target.ZonePosition, -2] = 1
                        elif option.Target.Zone.Controller.Name == self.game.CurrentOpponent.Name:
                            minion_scalar_batch[i, 7 + option.Target.ZonePosition, -2] = 1
                    elif option.Target == self.game.CurrentPlayer.Hero:
                        hero_scalar_batch[i, 0, -2] = 1
                    elif option.Target == self.game.CurrentOpponent.Hero:
                        hero_scalar_batch[i, 1, -2] = 1
                if option.ChooseOne in [1, 2]:
                    hand_card_scalar_batch[i, option.Source.ZonePosition, -2 - option.ChooseOne] = 1
            elif option_name == 'MinionAttackTask':
                minion_scalar_batch[i, option.Source.ZonePosition, -1] = 1
                if option.Target.Zone is not None:
                    if option.Target.Zone.Controller.Name == self.game.CurrentOpponent.Name:
                        minion_scalar_batch[i, 7 + option.Target.ZonePosition, -2] = 1
                elif option.Target == self.game.CurrentPlayer.Hero:
                    hero_scalar_batch[i, 0, -2] = 1
                elif option.Target == self.game.CurrentOpponent.Hero:
                    hero_scalar_batch[i, 1, -2] = 1
            elif option_name == 'HeroAttackTask':
                hero_scalar_batch[i, 0, -1] = 1
                if option.HasTarget:
                    if option.Target.Zone is not None:
                        if option.Target.Zone.Controller.Name == self.game.CurrentPlayer.Name:
                            minion_scalar_batch[i, option.Target.ZonePosition, -2] = 1
                        elif option.Target.Zone.Controller.Name == self.game.CurrentOpponent.Name:
                            minion_scalar_batch[i, 7 + option.Target.ZonePosition, -2] = 1
                    elif option.Target == self.game.CurrentPlayer.Hero:
                        hero_scalar_batch[i, 0, -2] = 1
                    elif option.Target == self.game.CurrentOpponent.Hero:
                        hero_scalar_batch[i, 1, -2] = 1
        obs = {
            "hand_card_names": game_state["hand_card_names"],
            "minion_names": game_state["minion_names"],
            "weapon_names": game_state["weapon_names"],
            "secret_names": game_state["secret_names"],
            "hand_card_scalar_batch": hand_card_scalar_batch,
            "minion_scalar_batch": minion_scalar_batch,
            "hero_scalar_batch": hero_scalar_batch,
        }
        return obs, options

    def get_next_state(self, current):
        next_state = self.get_current_state(current)
        next_state['minion_scalar'] = next_state['minion_scalar'][:,  3:12]
        next_state['hero_scalar'] = next_state['hero_scalar'][:, [1, 3,4,5,6,7,8,9,10,11,12,13,14,15,16,28]]

        return next_state


def validate_card(card_name):
    card = Cards.FromName(card_name)
    if card is not None and card.Implemented:
        return True
    else:
        return False
