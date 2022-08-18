from dataclasses import dataclass


@dataclass
class Configuration:
    """ Configuration used by this application. """
    # whenever an Optional value is None it means that this should value should be ignored a.k.a. open access
    host: str
    port: int
    secret: str
