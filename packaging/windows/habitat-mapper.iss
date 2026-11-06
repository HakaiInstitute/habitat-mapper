; Inno Setup script for the Habitat-Mapper Windows installer. Build after PyInstaller with:
;   iscc /DAppVersion=0.17.0 packaging\windows\habitat-mapper.iss

#ifndef AppVersion
  #define AppVersion "0.0.0"
#endif

[Setup]
AppId={{6F1E1B7A-3C5B-4B8E-9C61-2D8F0B7A4E15}
AppName=Habitat-Mapper
AppVersion={#AppVersion}
AppPublisher=Hakai Institute
AppPublisherURL=https://github.com/HakaiInstitute/habitat-mapper
AppSupportURL=https://habitat-mapper.readthedocs.io
; Installs for the current user, so it doesn't need administrator rights
PrivilegesRequired=lowest
DefaultDirName={localappdata}\Programs\Habitat-Mapper
DefaultGroupName=Habitat-Mapper
DisableProgramGroupPage=yes
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
OutputDir=..\..\dist
OutputBaseFilename=Habitat-Mapper-{#AppVersion}-windows-x64-setup
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
UninstallDisplayIcon={app}\Habitat-Mapper.exe

[Files]
Source: "..\..\dist\habitat-mapper\*"; DestDir: "{app}"; Flags: recursesubdirs createallsubdirs ignoreversion

[Icons]
Name: "{group}\Habitat-Mapper"; Filename: "{app}\Habitat-Mapper.exe"
Name: "{userdesktop}\Habitat-Mapper"; Filename: "{app}\Habitat-Mapper.exe"; Tasks: desktopicon

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; Flags: unchecked

[Run]
Filename: "{app}\Habitat-Mapper.exe"; Description: "Start Habitat-Mapper"; Flags: nowait postinstall skipifsilent
