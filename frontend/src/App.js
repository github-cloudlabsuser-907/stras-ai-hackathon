import React, { useState } from 'react';
import AppExplorer from './components/app_explorer/AppExplorer';
import { Typography, Container, Box, Button, Drawer, List, ListItem, ListItemText, Divider} from "@mui/material";
import KeyboardReturnIcon from '@mui/icons-material/KeyboardReturn';
import { DocumentManagerContextProvider } from './components/RefreshTab';
import './App.css';


function App() {
  const [selectedTab, setSelectedTab] = useState(0);
  const [isMenuOpen, setMenuOpen] = useState(false);

  const handleTabChange = (newValue) => {
    setSelectedTab(newValue);
    setMenuOpen(false); // Close the side menu when a tab is selected
  };

  return (
    <DocumentManagerContextProvider>
        <div className="App">
          {/* Title */}
          <Container sx={{ marginTop: '5px', display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
            <Typography variant="h6" sx={{ fontSize: '32px', fontFamily: 'CustomFont, Courier New, monospace', margin: 0 }}>
            EffiSearch
            </Typography>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center'}}>
              <Typography sx={{ fontSize: '14px', fontFamily: 'Courier New, monospace' }}>
                by
              </Typography>
              <img
                src={require('./static/logo_cap_white.png')}
                alt="Claudie Logo"
                style={{ width: '100px', height: 'auto', marginLeft: '5px' }}
              />
            </div>
          </Container>
          {/* Main Content */}
          <Container
            sx={{
              flex: 1, // Set 'flex' property to 1 to fill the available vertical space
              display: "flex",
              flexDirection: "column",
              alignItems: "center",
              justifyContent: "center",
              marginLeft: '20px', // Add left margin to create space for the vertical toolbar
            }}
          >
            <Container sx={{ display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center" }}>
              <div style={{ display: selectedTab === 0 ? 'block' : 'none' }}>
                <AppExplorer />
              </div>
              
            </Container>
          </Container>
        </div>
    </DocumentManagerContextProvider>
  );
}

export default App;
