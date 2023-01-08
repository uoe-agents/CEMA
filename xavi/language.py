import logging
from typing import Tuple, List

import numpy as np
import pandas as pd
import simplenlg as nlg
import re

from xavi.query import Query
from xavi.matching import ActionGroup

logger = logging.getLogger(__name__)


class Language:
    def __init__(self,
                 n_associative: int = 2,
                 n_final: int = 1,
                 n_efficient: Tuple[int, int] = (1, 4)):
        """ Initialise a new explanation generation language class.
        This class uses SimpleNLG by Gatt and Reiter, 2009 to generate explanations.

        Args:
            n_associative: The number of associative causes to use for explanations.
            n_final: The number of final causes to use for explanations.
            n_efficient: The number of efficient causes (per cause time) to use for explanations.
        """
        self.n_associative = n_associative
        self.n_final = n_final
        self.n_efficient = n_efficient

        self.__lexicon = nlg.Lexicon().getDefaultLexicon()
        self.__factory = nlg.NLGFactory(self.__lexicon)
        self.__realiser = nlg.Realiser(self.__lexicon)
        self.__tense_dict = {"past": nlg.Tense.PAST, "present": nlg.Tense.PRESENT, "future": nlg.Tense.FUTURE}

    def convert_to_sentence(self,
                            query: Query,
                            final_causes: pd.DataFrame,
                            efficient_causes: Tuple[pd.DataFrame, ...],
                            action_group: ActionGroup) -> (str, str):
        """ Convert a set of causal attributions to a natural language reply.

        Args:
            query: The query being answered.
            final_causes: Causal attributions for the final explanation.
            efficient_causes: Causal attributions for the efficient explanations.
            action_group: The action to be executed by the vehicle.

        Returns:
        """
        # Associative explanation
        associative_explanation = None
        if action_group is not None:
            associative_clause = self.__actiongroup_to_text(action_group, self.n_associative)
            associative_sentence = self.__factory.createClause("we", associative_clause)
            self.__set_tense(associative_sentence, query.tense)
            associative_explanation = self.__realiser.realiseSentence(associative_sentence)

        # Generate final explanation
        final_phrase = self.__reward_to_text(final_causes, self.n_final)
        final_sentence = self.__factory.createClause("it", final_phrase)
        self.__set_tense(final_sentence, query.tense)
        final_explanation = self.__realiser.realiseSentence(final_sentence)

        past_sents, future_sents = \
            self.__features_to_text(efficient_causes, self.n_efficient)
        efficient_paragraph = []
        for time, sents in [("past", past_sents), ("future", future_sents)]:
            for vid, phrase in sents:
                efficient_sentence = self.__factory.createClause(f"vehicle {vid}", phrase)
                self.__set_tense(efficient_sentence, query.tense, time)
                efficient_paragraph.append(self.__factory.createSentence(efficient_sentence))
        efficient_paragraph = self.__factory.createParagraph(efficient_paragraph)
        efficient_explanation = self.__realiser.realise(efficient_paragraph).getRealisation()
        return final_explanation, efficient_explanation, associative_explanation

    def __set_tense(self, phrase, tense: str, efficient: str = None):
        if efficient is not None:
            phrase.setTense(self.__tense_dict[tense])
            if efficient == "past":
                phrase.setFeature(nlg.Feature.PERFECT, True)
                if tense == "future":
                    phrase.setFeature(nlg.Feature.MODAL, "will")
        else:
            if tense == "past":
                phrase.setFeature(nlg.Feature.MODAL, "would")
                phrase.setFeature(nlg.Feature.PERFECT, True)
            elif tense == "present":
                phrase.setFeature(nlg.Feature.MODAL, "would")
            phrase.setTense(self.__tense_dict[tense])
            phrase.setFeature(nlg.Feature.AGGREGATE_AUXILIARY, True)

    def __actiongroup_to_text(self, action_group: ActionGroup, depth: int = 4) -> nlg.VPPhraseSpec:
        """ Return a VP representation of the counterfactual action.

        Args:
            action_group: The group of actions to detail
            depth: The number of actions to detail in the generated verb phrase
        """
        clause = self.__factory.createCoordinatedPhrase()
        clause.setFeature(nlg.Feature.CONJUNCTION, "then")
        clause.setFeature(nlg.Feature.AGGREGATE_AUXILIARY, True)

        d = 0
        for segment in action_group.segments:
            segment_clause = self.__factory.createCoordinatedPhrase()
            for action in reversed(segment.actions):
                segment_phrase = self.__action_to_verb(action)
                segment_clause.addCoordinate(segment_phrase)
                d += 1
                if d == depth: break
            clause.addCoordinate(segment_clause)
            if d == depth: break
        return clause

    def __action_to_verb(self, action: str) -> nlg.VPPhraseSpec:
        """ Convert an action from matching.py to a verb phrase. """
        camel_split = re.compile(r"[A-Z]?[a-z]+|[A-Z]+(?=[A-Z]|$)")
        man_split = camel_split.findall(action)
        man = " ".join(man_split[0:2]).lower()
        man_verb = self.__factory.createVerbPhrase(man)
        if len(man_split) > 2:
            if man == "change lane":
                man_verb.addComplement(f"to the {man_split[2].lower()}")
            elif man == "go straight":
                man_verb.addComplement(f"through the {man_split[2].lower()}")
        return man_verb

    def __reward_to_text(self, final_causes: pd.DataFrame, n_final: int = 1) -> nlg.VPPhraseSpec:
        """ Return a VP and NP representation of the reward change verb and object. """
        conversion_dict = {
            "time": "the time to reach the goal",
            "coll": "a collision",
            "angular_velocity": "lateral acceleration",
            "curvature": "curvature",
            "jerk": "jerk",
            "dead": "the goal"
        }
        pos_phrase = self.__factory.createCoordinatedPhrase()
        neg_phrase = self.__factory.createCoordinatedPhrase()
        phrase = self.__factory.createCoordinatedPhrase()
        for n in range(n_final):
            reward_type = final_causes.index[n]
            change = final_causes.loc[reward_type, "absolute"]
            object = self.__factory.createNounPhrase(conversion_dict.get(reward_type, "a change"))
            verb = self.__factory.createVerbPhrase("cause")
            negated = True
            if reward_type == "dead":
                verb = self.__factory.createVerbPhrase("reach")
                negated = change < 0
            elif reward_type != "coll":
                verb = self.__factory.createVerbPhrase("increase" if change > 0.0 else "decrease")
                negated = False
            verb.setNegated(negated)
            verb.setComplement(object)
            if negated:
                neg_phrase.addCoordinate(verb)
            else:
                pos_phrase.addCoordinate(verb)
        phrase.addCoordinate(pos_phrase)
        if "coordinates" in neg_phrase.features:
            phrase.setFeature(nlg.Feature.CONJUNCTION, "but")
            phrase.addCoordinate(neg_phrase)
        return phrase

    def __features_to_text(self,
                           efficient_causes: (pd.DataFrame, pd.DataFrame),
                           n_efficient: Tuple[int, int] = (1, 3)) \
            -> (List[Tuple[int, nlg.CoordinatedPhraseElement]], List[Tuple[int, nlg.CoordinatedPhraseElement]]):
        """ Convert the ordered list of efficient causes to a natural language sentence. """
        verbs_dict = {
            "samevelocity": ("have", "the same speed than us"),
            "exitright": ("turn", "right"),
            "exitleft": ("turn", "left"),
            "exitstraight": ("go", "straight"),
            "changelaneleft": ("change", "lane to the left"),
            "changelaneright": ("change", "lane to the right"),
            "continue": ("go", "straight"),
            "faster": ("be", "faster than us"),
            "slower": ("be", "slower than us"),
            "decelerate": ("slow down", None),
            "accelerate": ("speed up", None),
            "maintain": ("maintain", "velocity"),
            "stops": ("stop", None)
        }
        macro_re = re.compile(r"^(\w+)\(([^,]+)(,[^,]+)*\)$")

        def causes_to_verb(coef, n):
            c = 0
            coef = coef.mean(0)
            coef = coef[(-coef).argsort()]
            coef = coef[~np.isclose(coef, 0)]
            phrases = []
            coord = self.__factory.createCoordinatedPhrase()
            prev_vehicle_id = None
            for action, value in coef.iteritems():
                if not isinstance(action, str):
                    continue
                action_split = action.split("_")
                vehicle_id, action_split = action_split[0], action_split[1:]
                if "macro" in action_split:
                    action_split.remove("macro")
                    match = macro_re.match(action_split[0])
                    action, params = match.groups()[0].lower(), match.groups()[1:]
                    params = " ".join([p for p in params if p is not None and "[" not in p])
                    action = action + params
                else:
                    action = ''.join(action_split).lower()

                if action in verbs_dict:
                    verb, compl = verbs_dict[action]
                    verb = self.__factory.createVerbPhrase(verb)
                    if compl: verb.addComplement(compl)
                    c += 1
                else:
                    logger.debug(f"Unknown action key: {action}")
                    continue

                if prev_vehicle_id != vehicle_id and prev_vehicle_id is not None:
                    phrases.append((prev_vehicle_id, coord))
                    coord = self.__factory.createCoordinatedPhrase()
                clause = self.__factory.createClause()
                clause.setVerbPhrase(verb)
                coord.addCoordinate(clause)
                prev_vehicle_id = vehicle_id

                if c == n:
                    if "coordinates" in coord.features:
                        phrases.append((prev_vehicle_id, coord))
                    break
            return phrases

        past_causes, future_causes = efficient_causes
        past_verb = causes_to_verb(past_causes, n_efficient[0])
        future_verb = causes_to_verb(future_causes, n_efficient[1])
        return past_verb, future_verb
