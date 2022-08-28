# AI backend

The purpose of the backend is to update each active profile whenever the trial participant changes the setpoint temperature.

On a periodic basis the code runs the `check_for_changes` function. This code:
* checks for any unhandled setpoint changes;
* flags all setpoint changes to manual as checked;
* retrieves the profile that was active during the setpoint change;
* hands of the control to the AI component to update the profile.

> **Mark as Checked**<br>
> It is essential that, as soon as a profile has been updated, that the setpoint change that caused the profile update is marked as checked. This prevents this setpoint change from affecting the profile again in the next iteration. **Remember that the AI backend runs as a best effort; it is not guaranteed to work at the given intervals**.

For reasons if persistence the AI backend should be scheduled using `systemd` such that it is automatically started on boot and restarted on failure.