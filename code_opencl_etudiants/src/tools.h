#ifndef __TOOLS_H__
#define __TOOLS_H__

double top(int id = 0);

const char *getArg( const char *str, const char *argName);
const char *getArgValueFromCmdl( const char _argc, const char **_argv, const char *argName);

#endif
