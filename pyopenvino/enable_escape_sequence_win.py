from ctypes import windll, wintypes, byref
from functools import reduce

def enable():
  INVALID_HANDLE_VALUE = -1
  STD_INPUT_HANDLE = -10
  STD_OUTPUT_HANDLE = -11
  STD_ERROR_HANDLE = -12
  ENABLE_VIRTUAL_TERMINAL_PROCESSING = 0x0004
  ENABLE_LVB_GRID_WORLDWIDE = 0x0010

  hOut = windll.kernel32.GetStdHandle(STD_OUTPUT_HANDLE)
  if hOut == INVALID_HANDLE_VALUE:
    return False
  dwMode = wintypes.DWORD()
  if windll.kernel32.GetConsoleMode(hOut, byref(dwMode)) == 0:
    return False
  dwMode.value |= ENABLE_VIRTUAL_TERMINAL_PROCESSING
  # dwMode.value |= ENABLE_LVB_GRID_WORLDWIDE
  if windll.kernel32.SetConsoleMode(hOut, dwMode) == 0:
    return False
  return True

enable()
