#pragma once

#include "GlobalHelpers.h"
#include "CImplementationBase.h"

class CPlatformSelection {
 public:

 private:
  CImplementationBase *m_ptrImplCpu;
  CImplementationBase *m_ptrImplXil;

};

