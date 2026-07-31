describe('template spec', () => {
  it('passes', () => {
    cy.viewport(1200, 800)
    cy.visit('http://localhost:8080')
    cy.contains('Sign in').click()
    cy.get('#message').type('hello world {enter}')
    
  })
})