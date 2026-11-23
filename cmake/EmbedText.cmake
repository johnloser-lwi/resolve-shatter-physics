# Wraps a text file in a C++ raw string literal.
#
#   cmake -DIN=<file> -DOUT=<header> -DSYMBOL=<name> -P EmbedText.cmake
#
# Run as a BUILD step (add_custom_command), not at configure time. A header
# written during configure can land in the same second as the object compiled
# from it, and make then judges the object up to date: an edited shader would
# silently keep running the previous kernel, which is exactly the failure a
# parity test cannot see past.
file(READ "${IN}" TEXT)
string(FIND "${TEXT}" ")SHPEMBED" clash)
if(NOT clash EQUAL -1)
  message(FATAL_ERROR "${IN} contains the raw-string terminator )SHPEMBED")
endif()
file(WRITE "${OUT}" "// Generated from ${IN} -- edit that, not this.\n#pragma once\n")
file(APPEND "${OUT}" "static const char ${SYMBOL}[] = R\"SHPEMBED(")
file(APPEND "${OUT}" "${TEXT}")
file(APPEND "${OUT}" ")SHPEMBED\";\n")
