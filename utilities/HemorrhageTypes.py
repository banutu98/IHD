from enum import Enum


class HemorrhageTypes(Enum):
    EP = "epidural"
    IN_PA = "intraparenchymal"
    IN_VE = "intraventricular"
    SUB_AR = "subarachnoid"
    SUB_DU = "subdural"
    ANY = "any"
