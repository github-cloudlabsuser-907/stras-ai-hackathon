import React, { useState } from 'react';
import { Button, CircularProgress, Grid, Select, MenuItem, FormControl, InputLabel  } from '@mui/material';

const AppExplorer = () => {
  const apiBase = '/api';
  const [loading, setLoading] = useState(false);
  const [selectedDifficulty, setSelectedDifficulty] = useState('Beginner');
  const [selectedFiles, setSelectedFiles] = useState([]);

  const handleFileChange = (event) => {
    const newFiles = event.target.files;
  
    const excludedExtensions = ['.md', '.xml', '.json', '.xml', '.mvn', '.jpg', '.properties', '.jar', '.gitignore', '.cmd', 'mvnw', '.sql', '.pyc'];
    const excludedFilesAndFolders = ['excludeThisFile.txt', 'src/test'];
  
    const extractFilesFromFolder = (folder) => {
      let files = [];
      for (const file of folder) {
        if (file.isDirectory) {
          if (!excludedFilesAndFolders.includes(file.name)) {
            // Si c'est un sous-dossier non exclu, récursivement extraire les fichiers
            files = files.concat(extractFilesFromFolder(file));
          }
        } else {
          const fileExtension = file.name.split('.').pop().toLowerCase();
          const fileName = file.name.toLowerCase();
          if (
            !excludedExtensions.includes(`.${fileExtension}`) &&
            !excludedFilesAndFolders.includes(file.name) &&
            !fileName.includes('test')
          ) {
            files.push(file);
          }
        }
      }
      return files;
    };
  
    const selectedFiles = extractFilesFromFolder(newFiles);
  
    setSelectedFiles([...selectedFiles]);
  };
  

  const handleRemoveFile = (index) => {
    const newSelectedFiles = [...selectedFiles];
    newSelectedFiles.splice(index, 1);
    setSelectedFiles(newSelectedFiles);
  };

 const handleExplainCode = () => {
    setLoading(true);
    const formData = new FormData();
    selectedFiles.forEach((file) => {
      formData.append('files', file);
    });
    formData.append('difficulty', selectedDifficulty);

    // Le fetch reste le même, mais vous devez vous préparer à recevoir un blob au lieu d'un JSON.
    fetch(`${apiBase}/explain_application`, {
        method: 'POST',
        body: formData,
    })
    .then((response) => {
        if (response.ok) {
            // Vous devez renvoyer à la fois le blob et la réponse ici pour avoir accès aux en-têtes ensuite.
            return Promise.all([response.blob(), Promise.resolve(response)]);
        }
        throw new Error("Network response was not ok.");
    })
    .then(([blob, response]) => {
        // Maintenant, 'response' est disponible et vous pouvez extraire l'en-tête 'Content-Disposition'.
        const contentDisposition = response.headers.get('Content-Disposition');
        const fileName = contentDisposition ? contentDisposition.split('filename=')[1] : 'document.docx';

        // Nous créons directement l'URL et déclenchons le téléchargement.
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.style.display = 'none';
        a.href = url;
        a.download = fileName;
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);
        setLoading(false);
        setSelectedFiles([]);
    })
    .catch((error) => {
        console.error('Error explaining code:', error);
        setLoading(false);
    });
};

  const handleDrop = (event) => {
    event.preventDefault();
    const droppedFiles = event.dataTransfer.files;
    setSelectedFiles([...selectedFiles, ...droppedFiles]);
  };

  const handleDragOver = (event) => {
    event.preventDefault();
  };

  return (
    <Grid
      sx={{
        flexGrow: 1,
        position: 'fixed',
        bottom: 0,
        left: 0,
        right: 0,
        top: 0,
        height: 'calc(100% - 250px)',
        maxWidth: '84%',
        margin: 'auto',
        justifyContent: 'center',
        alignItems: 'center',
      }}
      container
      spacing={2}
      direction="row"
      alignItems="center"
      justifyContent="center"
    >
    <Grid item xs={9} md={6}>
      <div
        onDrop={handleDrop}
        onDragOver={handleDragOver}
        style={{
          border: '2px dashed #aaa',
          borderRadius: '4px',
          padding: '1rem',
          width: '100%',
          height: '30%',
          display: 'flex',
          flexDirection: 'column',
          justifyContent: 'center',
          alignItems: 'center',
          cursor: 'pointer',
        }}
      >
        <input
          type="file"
          directory=""
          webkitdirectory=""
          onChange={handleFileChange}
          style={{ display: 'none' }}
          id="file-input-explorer"
          multiple // Activez la sélection de plusieurs fichiers
        />
          <label htmlFor="file-input-explorer">
            <p>Submit your code folder for which you want to generate the explanation</p>
          </label>
        </div>
        {selectedFiles.length > 0 && (
          <div style={{ marginTop: '2%', maxHeight: '150px', overflowY: 'auto' }}>
            <p style={{ marginTop: '2%' }}>Fichiers sélectionnés :</p>
            <ul>
              {selectedFiles.map((file, index) => (
                <li key={index}>
                    {file.name}
                    <Button
                    variant="outlined"
                    color="secondary"
                    size="small"
                    onClick={() => handleRemoveFile(index)} // Appeler la fonction de suppression avec l'index
                    style={{ marginLeft: '8px' }}
                  >
                    Supprimer
                  </Button>
                </li>
              ))}
            </ul>
          </div>
        )}
      <div style={{ display: 'flex', flexDirection: 'row', alignItems: 'center', justifyContent: 'center' }}>
        <FormControl variant="outlined" style={{ flex: 1, marginTop: '4%', marginRight: '2%' }}>
          <InputLabel>Level of explanation</InputLabel>
          <Select
            value={selectedDifficulty}
            onChange={(e) => setSelectedDifficulty(e.target.value)}
            label="Level of explanation"
          >
            <MenuItem value="Beginner">Beginner</MenuItem>
            <MenuItem value="Advanced">Advanced</MenuItem>
            {/* Ajoutez d'autres niveaux de difficulté au besoin */}
            </Select>
        </FormControl>
      </div>
      <div style={{ marginTop: '2%', textAlign: 'center', width: '100%' }}>
        {loading ? (
          <CircularProgress />
        ) : (
          <Button variant="contained" onClick={handleExplainCode} disabled={!selectedFiles.length || loading}>
            Explain Code
          </Button>
        )}
      </div>
    </Grid>
  </Grid>
  );
};

export default AppExplorer;