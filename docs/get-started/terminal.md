# Using the Terminal

The terminal (or "Command Line") is a text-based way to talk to your computer. Instead of clicking icons, you type
commands to tell the computer what to do.

You will use the terminal to install Habitat-Mapper and run the image processing tools.

---

## 1. Opening the Terminal

Choose your operating system to see how to start.

=== "Windows"
    We recommend using **PowerShell** (not the old "Command Prompt").

    1. Press the **Windows Key**.
    2. Type `PowerShell`.
    3. Click **Windows PowerShell** to open it.

    *(It will look like a blue or black window with text waiting for you to type).*

=== "MacOS"
    1. Press ++command+space++ to open Spotlight Search.
    2. Type `Terminal`.
    3. Press ++enter++.

=== "Linux"
    You likely already know this, but usually:

    1. Press ++ctrl+alt+t++.
    2. Or search `Terminal` in your applications menu.

---

## 2. The "Magic" Keys :material-auto-fix:

Before typing commands, learn these two keys. They will save you hours of typing and frustration.

### Tab Completion (The most important key!)

Never type a full filename or folder name manually.
Instead, type the first few letters and press ++tab++. The terminal will auto-complete the name for you.

* **Why?** It prevents typos. If you press ++tab++ and nothing happens, you know you made a mistake or the file isn't
  there.

### The Up Arrow

Press the ++arrow-up++ key to see the last command you ran. Press it again to go back further.
This is great for re-running a command without re-typing it.

---

## 3. Basic Navigation

You can think of the terminal as being "inside" a specific folder on your computer. You use commands to look around or
move to a different folder.

### The Cheat Sheet

=== "Windows (PowerShell)"

    | Goal | Command | Example |
    | :--- | :--- | :--- |
    | **List** files in current folder | `ls` or `dir` | `ls` |
    | **Change** to a folder | `cd` | `cd Documents` |
    | **Go Back** one folder | `cd ..` | `cd ..` |
    | **Go Home** (User folder) | `cd ~` | `cd ~` |
    | **Where am I?** | `pwd` | `pwd` |

=== "MacOS / Linux"

    | Goal | Command | Example |
    | :--- | :--- | :--- |
    | **List** files in current folder | `ls` | `ls` |
    | **Change** to a folder | `cd` | `cd Documents` |
    | **Go Back** one folder | `cd ..` | `cd ..` |
    | **Go Home** (User folder) | `cd ~` | `cd ~` |
    | **Where am I?** | `pwd` | `pwd` |

### Understanding Paths

A "Path" is just an address for a file. You will often see special symbols used in these addresses.

#### The Dot Symbols (`.` and `..`)

These are shortcuts that act like relative directions:

* `.` **(One Dot) = "Here"** (The folder you are currently in).
    * Usage: You often see this when running scripts, like `./activate` (run the activate script located right here).
* `..` **(Two Dots) = "Parent Folder"** (The folder directly above this one).
    * Usage: Typing `cd ..` is like clicking the "Up" or "Back" button in your file explorer. It moves you out of the
      current folder.

#### Rules of the Road

* **Spaces:** If a folder has a space in the name, you **must** use quotes.
    * :x: `cd My Documents` (Will fail)
    * :white_check_mark: `cd "My Documents"` (Works!)
* **Separators:**
    * Windows uses backslashes `\` (e.g., `C:\Users\Name`).
    * Mac/Linux uses forward slashes `/` (e.g., `/home/name`).

!!! example "Try it yourself"
    Open your terminal and try these steps:

    1. Type `cd ~` and press ++enter++ (Goes to your home folder).
    2. Type `cd Do` and press ++tab++ (Should auto-complete to `Downloads` or `Documents`).
    3. Press ++enter++ to go there.
    4. Type `ls` (or `dir` on Windows) to see what is inside.

    **What success looks like:**

    You should see a list of files and folders in your terminal. The list might include folders like "Projects", images, or documents you've saved there.

    If you see an error like `No such file or directory`, press the ++arrow-up++ key to see your last command and check for typos. Remember to use ++tab++ completion to avoid mistakes!

---

## 4. Copying and Pasting

Pasting commands into a terminal can be tricky.

* **Windows (PowerShell):** usually **Right-Click** anywhere in the window to paste. (++ctrl+v++ works in newer
  versions).
* **Mac (Terminal):** use ++command+v++.
* **Linux:** usually ++ctrl+shift+v++ (since ++ctrl+v++ often has a different meaning).

---

## Next Steps

Now that you can move around, let's get the software installed.

[:material-arrow-right: Next: Install](install.md){ .md-button .md-button--primary }
