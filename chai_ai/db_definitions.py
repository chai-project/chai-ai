# pylint: disable=line-too-long, missing-module-docstring, too-few-public-methods, missing-class-docstring
# pylint: disable=singleton-comparison

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Optional

from sqlalchemy import Column, String, Integer, Float, DateTime, ForeignKey, TIMESTAMP, Index, JSON, Boolean
from sqlalchemy import and_
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import aliased
from sqlalchemy.orm import relationship, declarative_base
from sqlalchemy.orm import scoped_session, Session, sessionmaker


@dataclass
class Configuration:
    server: str
    username: str
    password: str
    database: str = "chai"
    enable_debugging: bool = False


def db_engine(config: Configuration) -> Engine:
    """
    Get a database engine.
    :param config: The configuration to use to initialise the database engine.
    :return: A database engine connection.
    """
    target = f"postgresql+pg8000://{config.username}:{config.password}@{config.server}/{config.database}"
    return create_engine(target, echo=config.enable_debugging, future=True, client_encoding="utf8")


@contextmanager
def db_engine_manager(config: Configuration) -> Engine:
    """
    A context manager that yields a database engine.
    :param config: The configuration to use to initialise the database engine.
    :return: A database engine connection.
    """
    yield db_engine(config)


@contextmanager
def db_session_manager(engine: Engine) -> Session:
    """
    A context manager that yields a database connection.
    :param engine: The database engine to bind the session to.
    :return: A database session.
    """
    _session = sessionmaker(engine)()
    try:
        yield _session
        _session.commit()
    finally:
        _session.close()


# database naming convention is to use camelCase and singular nouns throughout

Base = declarative_base()


class NetatmoDevice(Base):
    __tablename__ = "netatmodevice"
    id = Column(Integer, primary_key=True)
    refreshToken = Column("refreshtoken", String, nullable=False)
    readings = relationship("NetatmoReading", back_populates="relay")


class Home(Base):
    __tablename__ = "home"
    id = Column(Integer, primary_key=True)
    label = Column(String, nullable=False)
    revision = Column(TIMESTAMP, nullable=False)
    netatmoID = Column("netatmoid", Integer, ForeignKey("netatmodevice.id"), nullable=False)
    heat_gain = Column("heatgain", Float, nullable=False)
    heat_loss = Column("heatloss", Float, nullable=False)
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    relay: NetatmoDevice = relationship("NetatmoDevice")


class NetatmoReading(Base):
    __tablename__ = "netatmoreading"
    id = Column(Integer, primary_key=True)
    room_id = Column("roomid", Integer, nullable=False)  # 1 for thermostat temp, 2 for valve temp, 3 for valve %
    netatmo_id = Column("netatmoid", Integer, ForeignKey("netatmodevice.id"), nullable=False)
    start = Column(DateTime, nullable=False, index=True)
    end = Column(DateTime, nullable=False, index=True)
    reading = Column(Float, nullable=False)
    relay: NetatmoDevice = relationship("NetatmoDevice", back_populates="readings")
    idxOneReading = Index("ix_one_reading", id, room_id, start, unique=True)


class SetpointChange(Base):
    __tablename__ = "setpointchange"
    id = Column(Integer, primary_key=True)
    home_id = Column("homeid", Integer, ForeignKey("home.id"), nullable=False)
    changed_at = Column("changedat", DateTime, nullable=False)
    expires_at = Column("expiresat", DateTime, nullable=False)
    checked = Column(Boolean, nullable=False)
    duration = Column(Integer)
    mode = Column(Integer, nullable=False)
    temperature = Column(Float)
    home: Home = relationship("Home")


class Log(Base):
    __tablename__ = "log"
    id = Column(Integer, primary_key=True)
    home_id = Column("homeid", Integer, ForeignKey("home.id"), nullable=False)
    timestamp = Column(TIMESTAMP, nullable=False)
    category = Column(String, nullable=False)
    parameters = Column(JSON, nullable=False)


class Schedule(Base):
    __tablename__ = "schedule"
    id = Column(Integer, primary_key=True)
    home_id = Column("homeid", Integer, ForeignKey("home.id"), nullable=False)
    revision = Column(TIMESTAMP, nullable=False)
    day = Column(Integer, nullable=False)
    schedule = Column(JSON, nullable=False)
    home: Home = relationship("Home")


class Profile(Base):
    __tablename__ = "profile"
    id = Column(Integer, primary_key=True)
    profile_id = Column("profileid", Integer, nullable=False)  # has a value of 1 to 5 to indicate the profile
    home_id = Column("homeid", Integer, ForeignKey("home.id"), nullable=False)
    setpoint_id = Column("setpointid", Integer, ForeignKey("setpointchange.id"), nullable=False)
    mean1 = Column(Float, nullable=False)
    mean2 = Column(Float, nullable=False)
    variance1 = Column(Float, nullable=False)
    variance2 = Column(Float, nullable=False)
    correlation = Column(Float, nullable=False)
    noise_variance = Column("noisevariance", Float, nullable=False)
    home: Home = relationship("Home")
    setpointChange: SetpointChange = relationship("SetpointChange")

    def calculate_temperature(self, price: float):
        """
        Calculate the temperature for the given price for this model.
        :param price: The price used to calculate the temperature.
        :return: The temperature based on the model.
        """
        return price * self.mean2 + self.mean1


def get_home(label: str, session: Session) -> Optional[Home]:
    """
    Get the home associated with a given label.
    :param label: The label of the home to get.
    :param session: The database session to use.
    :return: The home associated with the label.
    """
    home_alias = aliased(Home)
    home = session.query(
        Home
    ).outerjoin(
        home_alias, and_(Home.label == home_alias.label, Home.revision < home_alias.revision)
    ).filter(
        home_alias.revision == None  # noqa: E711
    ).filter(
        Home.label == label
    ).first()
    return home


if __name__ == "__main__":
    pass
