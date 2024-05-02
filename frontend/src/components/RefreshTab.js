import React, { createContext, useContext, useState } from 'react';

const DocumentManagerContext = createContext();

export const useDocumentManagerContext = () => {
  return useContext(DocumentManagerContext);
};

export const DocumentManagerContextProvider = ({ children }) => {
  const [shouldRefresh, setShouldRefresh] = useState(false);

  const refreshDocumentManager = () => {
    setShouldRefresh(true);
  };

  const resetRefreshFlag = () => {
    setShouldRefresh(false);
  };

  return (
    <DocumentManagerContext.Provider value={{ shouldRefresh, refreshDocumentManager, resetRefreshFlag }}>
      {children}
    </DocumentManagerContext.Provider>
  );
};
