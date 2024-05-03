import React, { useState } from 'react';
import { Button, CircularProgress, Grid, TextField  } from '@mui/material';

const AppExplorer = () => {
  const apiBase = '/api';
  const [loading, setLoading] = useState(false);
  const [userQuestion, setUserQuestion] = useState("");
  const [response, setResponse] = useState("");

  const handleChangeUserQuestion = (event) => {
    setUserQuestion(event.target.value);
    console.log(event.target.value)
  };

  const handleAnswerQuestion = (event) => {
    setLoading(true);
    setResponse("");
    const formData = new FormData();
    formData.append('question', userQuestion);

    fetch(`${apiBase}/explain_question`, {
      method: 'POST',
      body: formData,
    })
    .then((response) => {
      if (response.ok) {
        return Promise.all([response.blob(), Promise.resolve(response)]);
      }
      throw new Error("Network response was not ok.");
    })
    .then(([blob, response]) => {
      const contentDisposition = response.headers.get('Content-Disposition');
      const filename = ""
    })
    .catch((error) => {
      console.error('Error asking question. Code: ', error);
      setLoading(false);
    })
  }

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
        // onDrop={handleDrop}
        // onDragOver={handleDragOver}
        style={{
          // border: '2px dashed #aaa',
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
        <TextField
          fullWidth
          multiline={true}
          rows={4}
          label="Please enter your question"
          value={userQuestion}
          onChange={handleChangeUserQuestion}/>
      </div>
      <div
        style={{padding: '1rem'}}
        >
        <TextField
          fullWidth
          multiline={true}
          rows={8}
          value={response}
          disabled={true}
          />
      </div>
      <div style={{ marginTop: '2%', textAlign: 'center', width: '100%' }}>
        {loading ? (
          <CircularProgress />
        ) : (
          <Button variant="contained" onClick={handleAnswerQuestion} disabled={userQuestion.length === 0 || loading}>
            Ask question
          </Button>
        )}
      </div>
    </Grid>
  </Grid>
  );
};

export default AppExplorer;