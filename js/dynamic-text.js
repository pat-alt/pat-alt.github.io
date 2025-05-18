// dynamic-text.js

document.addEventListener('DOMContentLoaded', function() {
  // Get sentences from the div content
  const dynamicTextElement = document.querySelector(".dynamic-text");
  
  // Extract sentences from the div text content
  const rawText = dynamicTextElement.textContent.trim();
  const sentences = rawText.split('\n')
                           .map(sentence => sentence.trim())
                           .filter(sentence => sentence.length > 0);
  
  console.log("Extracted sentences:", sentences);
  
  // Clear the original content
  dynamicTextElement.textContent = '';
  
  let currentIndex = 0;
  
  // Create a span element for the text to confine gradient to text width
  const textSpan = document.createElement('span');
  textSpan.className = 'gradient-text';
  dynamicTextElement.appendChild(textSpan);
  
  // Create cursor element
  const cursorSpan = document.createElement('span');
  cursorSpan.className = 'typing-cursor';
  cursorSpan.textContent = '|';
  dynamicTextElement.appendChild(cursorSpan);
  
  // Function to simulate typing animation
  function typeWriter(text, i, fnCallback) {
    // Check if we've reached the end of the text
    if (i < text.length) {
      // Add next character to text span
      textSpan.textContent = text.substring(0, i+1);
      
      // Wait for a random amount of time (between 50-150ms) to simulate human typing
      setTimeout(function() {
        typeWriter(text, i + 1, fnCallback)
      }, Math.random() * 100 + 50);
    }
    // If we've reached the end of the text, wait and then call callback
    else if (typeof fnCallback == 'function') {
      // Wait 2 seconds then start erasing
      setTimeout(fnCallback, 2000);
    }
  }
  
  // Function to erase text
  function eraseText(text, i, fnCallback) {
    if (i >= 0) {
      // Remove one character
      textSpan.textContent = text.substring(0, i);
      
      // Wait for a random amount of time to simulate human erasing
      setTimeout(function() {
        eraseText(text, i - 1, fnCallback)
      }, Math.random() * 50 + 30);
    } else if (typeof fnCallback == 'function') {
      setTimeout(fnCallback, 500); // Wait half a second before typing next sentence
    }
  }
  
  // Function to start the typing animation cycle
  function startTyping() {
    const currentSentence = sentences[currentIndex];
    
    // Type the current sentence
    typeWriter(currentSentence, 0, function() {
      // After typing, erase the sentence
      eraseText(currentSentence, currentSentence.length, function() {
        // Move to next sentence
        currentIndex = (currentIndex + 1) % sentences.length;
        // Start typing the next sentence
        startTyping();
      });
    });
  }
  
  // Start the animation cycle
  startTyping();
  
  // Add blinking cursor animation
  setInterval(function() {
    cursorSpan.style.opacity = (cursorSpan.style.opacity === '0' ? '1' : '0');
  }, 500);
});
