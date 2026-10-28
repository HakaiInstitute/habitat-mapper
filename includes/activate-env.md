=== "Windows"

    ```powershell
    .\habitat-env\Scripts\activate
    ```

    !!! warning "Seeing *running scripts is disabled on this system*?"
        Run this once, then try the activation command again:

        ```powershell
        Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
        ```

=== "macOS / Linux"

    ```bash
    source ./habitat-env/bin/activate
    ```

Your terminal prompt should now start with `(habitat-env)`. You need to run this every time you open a
new terminal window.
