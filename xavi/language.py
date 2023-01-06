from typing import Tuple

import numpy as np
import pandas as pd
import simplenlg as nlg

from xavi.query import Query
from xavi.matching import ActionGroup


class Language:
    def __init__(self):
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
        if action_group is not None:
            associative_sentence = self.__factory.createClause()


        # Generate final explanation
        cause_type = final_causes.index[0]
        cause_verb, cause_object = self.__reward_to_text(
            cause_type, final_causes.loc[cause_type, "absolute"])
        final_sentence = self.__factory.createClause()
        cause_subject = self.__factory.createNounPhrase("it")
        final_sentence.setSubject(cause_subject)
        final_sentence.setVerbPhrase(cause_verb)
        final_sentence.setObject(cause_object)
        if query.tense == "past":
            cause_verb.setFeature(nlg.Feature.MODAL, "would")
            cause_verb.setFeature(nlg.Feature.PERFECT, True)
        elif query.tense == "present":
            cause_verb.setFeature(nlg.Feature.MODAL, "would")
        cause_verb.setTense(self.__tense_dict[query.tense])
        final_explanation = self.__realiser.realiseSentence(final_sentence)

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
        return final_explanation, efficient_explanation

    def __reward_to_text(self, reward_type: str, change: float) -> (nlg.VPPhraseSpec, nlg.NPPhraseSpec):
        """ Return a textual representation of the reward change. """
        # clause = self.__factory.createClause()
        # subject = self.__factory.createNounPhrase("time")
        # verb = self.__factory.createVerbPhrase("reach")
        # object = self.__factory.createNounPhrase("goal")
        # object.setDeterminer("the")
        # verb.setFeature(nlg.Feature.FORM, nlg.Form.INFINITIVE)
        # clause.setSubject(subject)
        # clause.setVerb(verb)
        # clause.setObject(object)
        # sent = self.__realiser.realiseSentence(clause)

        object = self.__factory.createNounPhrase({
            "time": "the time to reach the goal",
            "coll": "a collision",
            "angular_velocity": "lateral acceleration",
            "curvature": "curvature",
            "jerk": "jerk"
        }.get(reward_type, ""))

        verb = self.__factory.createVerbPhrase("cause")
        if np.isclose(change, 0.0):
            verb.setNegated(True)
            object = self.__factory.createNounPhrase("a change")
        elif reward_type != "coll":
            verb = self.__factory.createVerbPhrase("increase" if change > 0.0 else "decrease")
        return verb, object
