from abc import ABC

class BaseClass(ABC):
    _ataque = 0
    _defensa = 0


    def __init__(self, ataque_base, defensa_base, performance):
        self._ataque = ataque_base
        self._defensa = defensa_base
        self._performance = performance
        

class Guerrero(BaseClass):

    def __init__(self, ataque, defensa, performance):
        super().__init__(ataque,defensa, performance)



class Arquero(BaseClass):

    def __init__(self, ataque, defensa, performance):
        super().__init__(ataque,defensa, performance)



class Defensor(BaseClass):

    def __init__(self, ataque, defensa, performance):
        super().__init__(ataque,defensa, performance)


class Infiltrado(BaseClass):

    def __init__(self, ataque, defensa, performance):
        super().__init__(ataque,defensa, performance)