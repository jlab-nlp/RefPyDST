import abc

from refpydst.data_types import MultiWOZDict


class AbstractNormalizer(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def normalize(self, raw_parse: MultiWOZDict) -> MultiWOZDict:
        """
        Normalizer addresses issues like typos in a candidate parse. The general pipeline goes like:

        1. given: completion (string) from a model
        2. raw_parse = parse(completion) -> MultiWOZDict (initial parse based on completion string)
        3. normalized_parse = normalize(raw_parse) -> MultiWOZDict (a parse that is ready for system use/auto eval)

        This is an interface for defining different approaches to step 3

        :param raw_parse: MultiWOZDict containing potentially un-normalized slot values
        :return: normalized dictionary ready for system use/eval
        """
        pass
