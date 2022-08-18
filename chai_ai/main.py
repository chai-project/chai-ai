import os
import sys
import time

import click
import pendulum
import schedule
import tomli
from sqlalchemy import and_
from sqlalchemy.orm import aliased

from db_definitions import SetpointChange, db_engine_manager, db_session_manager, Configuration as DBConfiguration, \
    Schedule, Profile


class Configuration:  # pylint: disable=too-few-public-methods, too-many-instance-attributes
    """ Configuration used by the AI instance. """
    db_server: str = "127.0.0.1"
    db_name: str = "chai"
    db_username: str = ""
    db_password: str = ""
    db_debug: bool = False

    def __str__(self):
        return (f"Configuration(db_server={self.db_server}, db_name={self.db_name}, "
                f"db_username={self.db_username}, db_password={self.db_password}, "
                f"db_debug={self.db_debug})")


# This is the main block of code where checks happen.
# It is called from the main() function, which also specifies how often this function is triggered.
def check_for_changes(config: DBConfiguration):
    # see which setpoint changes have occurred
    with db_engine_manager(config) as engine:
        with db_session_manager(engine) as session:
            changes = session.query(
                SetpointChange
            ).filter(
                SetpointChange.checked == False  # noqa: E711
            ).all()

            # the variable 'changes' contains all setpoint changes that still need to be checked

            changes_to_manual = [change for change in changes if change.mode in [2, 3]]
            # manual overrides do not affect the AI, marked them as checked
            for change in changes_to_manual:
                change.checked = True

            # what remains are all the setpoint changes in auto mode
            changes: [SetpointChange] = [change for change in changes if change.mode == 1]

            # we need to find out which profile was active at the time of the setpoint change
            for change in changes:
                changed_at = pendulum.instance(change.changed_at)
                daymask = 2 ** (changed_at.day_of_week - 1)

                # get the last schedule for this home with a timestamp before the setpoint change
                # TODO: this isn't working! The query should only perform the outer join with entries before change.changed_at .
                schedule_alias = aliased(Schedule)
                schedules = session.query(
                    Schedule
                ).outerjoin(
                    schedule_alias, and_(
                        Schedule.home_id == schedule_alias.home_id,
                        Schedule.revision < schedule_alias.revision,
                        Schedule.day == schedule_alias.day)
                ).filter(
                    schedule_alias.revision == None  # noqa: E711
                ).filter(
                    Schedule.home_id == change.home_id
                ).filter(
                    Schedule.day.op('&')(daymask) != 0
                ).all()

                if len(schedules) != 1:
                    # something went wrong as we expect only a single result
                    continue

                schedule = schedules[0]

                # this gives us the schedule entry
                profiles_schedule = [(int(slot), int(profile)) for slot, profile in schedule.schedule.items()]
                profiles_schedule.sort(key=lambda x: x[0])
                profiles_schedule.reverse()
                # this is a sorted list of slots at which a different profile triggers, e.g. [(40, 2), (32, 1), (24, 2)]

                # calculate at which slot the setpoint change occurred
                slot = changed_at.hour * 4 + changed_at.minute // 15
                # find the profile where the slot value is less than this slot
                profile_id = next(filter(lambda entry: entry[0] <= slot, profiles_schedule), None)

                profiles = session.query(
                    Profile
                ).filter(
                    Profile.homeid == change.home_id
                ).filter(
                    Profile.profileid == profile_id
                ).all()

                if len(profiles) != 1:
                    # something went wrong as we expect only a single result
                    continue

                profile = profiles[0]

                # AI COMPONENT
                # This is where the AI takes over. Note that the code above:
                #  - still has an error as it does not take the changed_at into account when retrieving the schedule
                #  - ignores that multiple setpoint changes may have occurred already for the same profile
                #  - is not fully tested

                pass

                # /AI COMPONENT

            # the manager automatically commits all changes here


def main(settings: Configuration):
    db_config = DBConfiguration(
        server=settings.db_server,
        username=settings.db_username,
        password=settings.db_password,
        database=settings.db_name,
        enable_debugging=settings.db_debug
    )

    check_for_changes(db_config)
    schedule.every(5).minutes.do(check_for_changes, config=db_config)

    while True:
        schedule.run_pending()
        time.sleep(60)


@click.command()
@click.option("--config", default=None, help="The TOML configuration file.")
@click.option("--dbserver", default=None, help="The server location of the PostgreSQL database, defaults to 127.0.0.1.")
@click.option("--db", default=None, help="The name of the database to access, defaults to chai.")
@click.option("--username", default=None, help="The username to access the database.")
@click.option("--dbpass_file", default=None, help="The file containing the (single line) password for database access.")
@click.option('--debug', is_flag=True, help="Provides debug output for the AI instance and the database when present.")
def cli(config, dbserver, db, username, dbpass_file, debug):  # pylint: disable=invalid-name
    settings = Configuration()

    if config and not os.path.isfile(config):
        click.echo("The configuration file is not found. Please provide a valid file path.")
        sys.exit(0)

    if config:
        with open(config, "rb") as file:
            try:
                toml = tomli.load(file)

                if toml_db := toml["database"]:
                    settings.db_server = str(toml_db.get("server", settings.db_server))
                    settings.db_name = str(toml_db.get("name", settings.db_name))
                    settings.db_username = str(toml_db.get("user", settings.db_username))
                    settings.db_password = str(toml_db.get("pass", settings.db_password))
                    settings.db_debug = bool(toml_db.get("debug", settings.db_debug))
            except tomli.TOMLDecodeError:
                click.echo("The configuration file is not valid and cannot be parsed.")
                sys.exit(0)

    # some entries may not be present in the TOML file, or they may be overridden by explicit CLI arguments

    # [overridden/supplemental database settings]
    if dbserver is not None:
        settings.db_server = dbserver

    if db is not None:
        settings.db_name = db

    if username is not None:
        settings.db_username = username

    # verify that the password file exists
    if dbpass_file and not os.path.isfile(dbpass_file):
        click.echo("Password file not found. Please provide a valid file path.")
        sys.exit(0)

    if dbpass_file:
        # use the contents of the file as the bearer token
        with open(dbpass_file, encoding="utf-8") as file:
            password = file.read().strip()
            settings.db_password = password

    if debug is True:
        settings.db_debug = True

    main(settings)


if __name__ == "__main__":
    # cli()
    # for testing in IDE:
    cli.callback(
        "/Users/kimbauters/Library/Mobile Documents/com~apple~CloudDocs/CHAI/Programming/Python/CHAI_ai/settings.toml",
        None, None, None, None, None)
