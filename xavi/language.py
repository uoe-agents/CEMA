from typing import Tuple

import numpy as np
import pandas as pd
import simplenlg as nlg
import re

from xavi.query import Query
from xavi.matching import ActionGroup


class Language:
    def __init__(self, n_associative: int = 2, n_final: int = 1, n_efficient: int = 2):
        """ Initialise a new explanation generation language class.
        This class uses SimpleNLG by Gatt and Reiter, 2009 to generate explanations.

        Args:
            n_associative: The number of associative causes to use for explanations.
            n_final: The number of final causes to use for explanations.
            n_efficient: The number of efficient causes to use for explanations.
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
            subject = self.__factory.createNounPhrase("we")
            associative_sentence = self.__factory.createClause()
            self.__set_tense(associative_sentence, query.tense)
            associative_sentence.setVerb(associative_clause)
            associative_sentence.setSubject(subject)
            associative_explanation = self.__realiser.realiseSentence(associative_sentence)

        # Generate final explanation
        final_phrase = self.__reward_to_text(final_causes, self.n_final)
        final_sentence = self.__factory.createClause()
        cause_subject = self.__factory.createNounPhrase("it")
        final_sentence.setSubject(cause_subject)
        final_sentence.setVerbPhrase(final_phrase)
        self.__set_tense(final_sentence, query.tense)
        final_explanation = self.__realiser.realiseSentence(final_sentence)

        efficient_phrase = self.__features_to_text(efficient_causes, self.n_efficient)
        efficient_sentence = self.__factory.createClause()
        cause_subject = self.__factory.createNounPhrase("Balint")
        verb = self.__factory.createVerbPhrase("will")
        object = self.__factory.createNounPhrase("apple")
        verb.addPreModifier("gluttonously")
        verb.setTense(nlg.Tense.PAST)
        object.setPlural(True)
        cause_subject.addModifier("hungry")
        efficient_sentence.setSubject(cause_subject)
        efficient_sentence.setVerbPhrase(verb)
        efficient_sentence.setObject(object)
        efficient_explanation = self.__realiser.realiseSentence(efficient_sentence)
        return final_explanation, efficient_explanation, associative_explanation

    def __set_tense(self, phrase, tense: str):
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

    def __features_to_text(self, efficient_causes: pd.DataFrame, n_efficient: int = 3) -> nlg.VPPhraseSpec:
        """ Convert the ordered list of efficient causes to a natural language sentence. """
        