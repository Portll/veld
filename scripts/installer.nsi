; Veld NSIS Installer
; Build: makensis scripts\installer.nsi
; Requires: NSIS 3.x (https://nsis.sourceforge.io)
;
; Before building, place these files in build\windows\:
;   veld.exe
;   veld-mcp.exe

!include "MUI2.nsh"
!include "FileFunc.nsh"
!include "LogicLib.nsh"

; ─── Product info ───────────────────────────────────────────────────────────
!define PRODUCT_NAME "Veld"
!define PRODUCT_PUBLISHER "Portll"
!define PRODUCT_WEB_SITE "https://github.com/Portll/veld"
!define PRODUCT_UNINST_KEY "Software\Microsoft\Windows\CurrentVersion\Uninstall\${PRODUCT_NAME}"
!define PRODUCT_UNINST_ROOT_KEY "HKCU"

; Version is injected by CI: makensis /DPRODUCT_VERSION=0.7.7
!ifndef PRODUCT_VERSION
  !define PRODUCT_VERSION "0.0.0"
!endif

; ─── Installer attributes ──────────────────────────────────────────────────
Name "${PRODUCT_NAME} ${PRODUCT_VERSION}"
OutFile "veld-${PRODUCT_VERSION}-x86_64-windows-setup.exe"
InstallDir "$LOCALAPPDATA\Veld"
RequestExecutionLevel user  ; No admin required
ShowInstDetails show
ShowUnInstDetails show

; ─── MUI settings ───────────────────────────────────────────────────────────
!define MUI_ABORTWARNING
!define MUI_WELCOMEPAGE_TITLE "Welcome to Veld ${PRODUCT_VERSION}"
!define MUI_WELCOMEPAGE_TEXT "Veld is a persistent cognitive memory system for AI agents.$\r$\n$\r$\nThis will install the Veld CLI and MCP server to your computer.$\r$\n$\r$\nClick Next to continue."
!define MUI_FINISHPAGE_RUN "$INSTDIR\bin\veld.exe"
!define MUI_FINISHPAGE_RUN_PARAMETERS "init"
!define MUI_FINISHPAGE_RUN_TEXT "Run first-time setup (veld init)"
!define MUI_FINISHPAGE_SHOWREADME ""
!define MUI_FINISHPAGE_SHOWREADME_TEXT "Add to PATH"
!define MUI_FINISHPAGE_SHOWREADME_FUNCTION AddToPath

; ─── Pages ──────────────────────────────────────────────────────────────────
!insertmacro MUI_PAGE_WELCOME
!insertmacro MUI_PAGE_LICENSE "..\..\LICENSE"
!insertmacro MUI_PAGE_DIRECTORY
!insertmacro MUI_PAGE_INSTFILES
!insertmacro MUI_PAGE_FINISH

!insertmacro MUI_UNPAGE_CONFIRM
!insertmacro MUI_UNPAGE_INSTFILES

!insertmacro MUI_LANGUAGE "English"

; ─── Install section ────────────────────────────────────────────────────────
Section "Veld Core" SEC_CORE
    SectionIn RO  ; Required section

    SetOutPath "$INSTDIR\bin"

    ; Copy binaries — CI places these in build\windows\ before running makensis
    File "..\..\build\windows\veld.exe"
    File "..\..\build\windows\veld-mcp.exe"

    ; Create uninstaller
    SetOutPath "$INSTDIR"
    WriteUninstaller "$INSTDIR\uninstall.exe"

    ; Start Menu shortcuts
    CreateDirectory "$SMPROGRAMS\${PRODUCT_NAME}"
    CreateShortCut "$SMPROGRAMS\${PRODUCT_NAME}\Veld Server.lnk" "$INSTDIR\bin\veld.exe" "server" "$INSTDIR\bin\veld.exe"
    CreateShortCut "$SMPROGRAMS\${PRODUCT_NAME}\Veld TUI.lnk" "$INSTDIR\bin\veld.exe" "tui" "$INSTDIR\bin\veld.exe"
    CreateShortCut "$SMPROGRAMS\${PRODUCT_NAME}\Uninstall.lnk" "$INSTDIR\uninstall.exe"

    ; Registry — Add/Remove Programs entry
    WriteRegStr ${PRODUCT_UNINST_ROOT_KEY} "${PRODUCT_UNINST_KEY}" "DisplayName" "${PRODUCT_NAME}"
    WriteRegStr ${PRODUCT_UNINST_ROOT_KEY} "${PRODUCT_UNINST_KEY}" "DisplayVersion" "${PRODUCT_VERSION}"
    WriteRegStr ${PRODUCT_UNINST_ROOT_KEY} "${PRODUCT_UNINST_KEY}" "Publisher" "${PRODUCT_PUBLISHER}"
    WriteRegStr ${PRODUCT_UNINST_ROOT_KEY} "${PRODUCT_UNINST_KEY}" "URLInfoAbout" "${PRODUCT_WEB_SITE}"
    WriteRegStr ${PRODUCT_UNINST_ROOT_KEY} "${PRODUCT_UNINST_KEY}" "UninstallString" "$INSTDIR\uninstall.exe"
    WriteRegStr ${PRODUCT_UNINST_ROOT_KEY} "${PRODUCT_UNINST_KEY}" "InstallLocation" "$INSTDIR"
    WriteRegDWORD ${PRODUCT_UNINST_ROOT_KEY} "${PRODUCT_UNINST_KEY}" "NoModify" 1
    WriteRegDWORD ${PRODUCT_UNINST_ROOT_KEY} "${PRODUCT_UNINST_KEY}" "NoRepair" 1

    ; Calculate installed size
    ${GetSize} "$INSTDIR" "/S=0K" $0 $1 $2
    IntFmt $0 "0x%08X" $0
    WriteRegDWORD ${PRODUCT_UNINST_ROOT_KEY} "${PRODUCT_UNINST_KEY}" "EstimatedSize" "$0"
SectionEnd

; ─── Add to PATH (user-level, no admin) ─────────────────────────────────────
Function AddToPath
    ; Read current user PATH
    ReadRegStr $0 HKCU "Environment" "Path"
    ${If} $0 == ""
        WriteRegExpandStr HKCU "Environment" "Path" "$INSTDIR\bin"
    ${Else}
        ; Check if already present
        StrCpy $1 "$0"
        ${WordFind} "$1" "$INSTDIR\bin" "E+1{" $2
        IfErrors 0 PathAlreadySet
            WriteRegExpandStr HKCU "Environment" "Path" "$0;$INSTDIR\bin"
        PathAlreadySet:
    ${EndIf}
    ; Broadcast WM_SETTINGCHANGE so running programs pick up the new PATH
    SendMessage ${HWND_BROADCAST} ${WM_SETTINGCHANGE} 0 "STR:Environment" /TIMEOUT=5000
FunctionEnd

; ─── Remove from PATH ───────────────────────────────────────────────────────
Function un.RemoveFromPath
    ReadRegStr $0 HKCU "Environment" "Path"
    ${If} $0 != ""
        ; Remove our directory from PATH string
        StrCpy $1 ""
        StrCpy $2 "$0"
        ${WordFind} "$2" ";" "E+1}" $3
        ${DoWhile} $3 != ""
            ${If} $3 != "$INSTDIR\bin"
                ${If} $1 == ""
                    StrCpy $1 "$3"
                ${Else}
                    StrCpy $1 "$1;$3"
                ${EndIf}
            ${EndIf}
            ${WordFind} "$2" ";" "E+1}" $3
        ${Loop}
        WriteRegExpandStr HKCU "Environment" "Path" "$1"
        SendMessage ${HWND_BROADCAST} ${WM_SETTINGCHANGE} 0 "STR:Environment" /TIMEOUT=5000
    ${EndIf}
FunctionEnd

; ─── Uninstall section ──────────────────────────────────────────────────────
Section "Uninstall"
    ; Remove binaries
    Delete "$INSTDIR\bin\veld.exe"
    Delete "$INSTDIR\bin\veld-mcp.exe"
    RMDir "$INSTDIR\bin"

    ; Remove Start Menu
    Delete "$SMPROGRAMS\${PRODUCT_NAME}\Veld Server.lnk"
    Delete "$SMPROGRAMS\${PRODUCT_NAME}\Veld TUI.lnk"
    Delete "$SMPROGRAMS\${PRODUCT_NAME}\Uninstall.lnk"
    RMDir "$SMPROGRAMS\${PRODUCT_NAME}"

    ; Remove uninstaller and install dir
    Delete "$INSTDIR\uninstall.exe"
    RMDir "$INSTDIR"

    ; Remove from PATH
    Call un.RemoveFromPath

    ; Remove registry
    DeleteRegKey ${PRODUCT_UNINST_ROOT_KEY} "${PRODUCT_UNINST_KEY}"
SectionEnd
