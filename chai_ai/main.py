import os
import sys
import time

import click
import pendulum
import schedule
import tomli
from sqlalchemy import and_
from sqlalchemy.orm import aliased

from chai_ai.learning import model_update, model_summary, model_predict
from db_definitions import SetpointChange, db_engine_manager, db_session_manager, Configuration as DBConfiguration, \
    Schedule, Profile, Log


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

            # see if there are any further changes to handle
            pending_changes = True
            while pending_changes:
                change = session.query(
                    SetpointChange
                ).filter(
                    SetpointChange.checked == False  # noqa: E711
                ).order_by(
                    SetpointChange.changed_at.asc()
                ).first()

                # get out of the loop if we are done with all changes
                if change is None:
                    pending_changes = False
                    continue

                # manual overrides do not affect the AI, marked them as checked and continue
                if change.mode in [2, 3]:
                    change.checked = True
                    continue

                # the instance 'change' is a setpoint change in auto mode that still needs to be checked by the AI code
                assert(change.mode == 1)

                # find out which profile was active at the time of the setpoint change
                changed_at = pendulum.instance(change.changed_at)
                daymask = 2 ** (changed_at.day_of_week - 1)

                # get the last schedule for this home with a timestamp before the setpoint change
                # BE CAREFUL WITH THIS QUERY:
                # We need to create a subquery and use that as a table.
                # However, SQLAlchemy cannot do any 'type checking' and instead relies on the columns from the DB.
                # We use the modifier .c to access those, but that also means some names, such as home_id, change.

                subquery = session.query(Schedule).filter(Schedule.revision <= change.changed_at).subquery()
                schedule_alias = aliased(subquery)

                schedules = session.query(
                    subquery
                ).outerjoin(
                    schedule_alias, and_(
                        subquery.c.homeid == schedule_alias.c.homeid,
                        subquery.c.revision < schedule_alias.c.revision,
                        subquery.c.day == schedule_alias.c.day)
                ).filter(
                    schedule_alias.c.revision == None  # noqa: E711
                ).filter(
                    subquery.c.homeid == change.home_id
                ).filter(
                    subquery.c.day.op('&')(daymask) != 0
                ).all()

                # we expect exactly one result
                assert(len(schedules) == 1)

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

                profile = session.query(
                    Profile
                ).filter(
                    Profile.home_id == change.home_id
                ).filter(
                    Profile.profile_id == profile_id[1]
                ).order_by(
                    Profile.id.desc()
                ).first()

                assert(profile is not None)

                # AI COMPONENT
                # This is where the AI takes over. Note that the code above:
                #  - ignores that multiple setpoint changes may have occurred already for the same profile
                #  - is not fully tested

                pass

                # change_price = 17.5  # To do: price needs to come from DB
                # updated_profile = model_update(profile, change, change_price)
                #
                # session.add(updated_profile)
                #
                # _, confidence_region = model_summary(updated_profile)
                # predictions = list(model_predict(updated_profile, range(36)))
                #
                # log_entry = Log(
                #     home=profile.home,
                #     timestamp=change.changed_at,
                #     category="PROFILE_CHANGE",
                #     parameters=[
                #         change.price,
                #         change.temperature,
                #         updated_profile.mean1,
                #         updated_profile.mean2,
                #         confidence_region,  # Note: may need to expand this tuple
                #         predictions  # Note: may need to expand this list of tuples
                #     ]
                # )
                #
                # session.add(log_entry)
                #
                # change.checked = True

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
    cli.callback("settings.toml", None, None, None, None, None)
