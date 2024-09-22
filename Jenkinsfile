pipeline {
    agent any
    
    stages {
        stage('Build') {
            steps {
                echo 'Building the application...'
                // Create a zip artifact of the entire project
                sh 'zip -r build_artifact.zip .'
            }
        }
        
        stage('Unit and Integration Tests') { // Stage 2
            steps {
                echo 'Integrating different tests...'
                sh '''
                # Activate virtual environment
                . /var/jenkins_home/workspace/project/venv/bin/activate
                # Run pytest inside the virtual environment
                pytest test_program.py
                '''
            }
        }
        
        stage('Code Analysis') {
            steps {
                echo 'Running SonarQube analysis for Python...'
        withSonarQubeEnv('My SonarQube') { // Make sure 'My SonarQube' is the name of your SonarQube configuration
            sh '''
            /opt/sonar-scanner/bin/sonar-scanner \
            -Dsonar.projectKey=my-python-project \
            -Dsonar.sources=. \
            -Dsonar.language=py \
            -Dsonar.host.url=http://sonarqube:9000/ \
            -Dsonar.login=squ_3721826b97039d12d511be94e0ba7f437cedfe53
            '''
                }
            }
        }
        
        stage('Deploy to Staging') {
            steps {
                echo 'Deploying to staging environment...'
                sh '''
                docker-compose down
                docker-compose up -d
                '''
            }
        }
        
        stage('Integration Tests on Staging') {
            steps {
                echo 'Running integration tests on staging...'
                // Example: sh 'mvn verify -Pstaging'
            }
        }
        
        stage('Deploy to Production') {
            steps {
                echo 'Deploying to production...'
                // Example: sh 'scp target/myapp.war user@production-server:/path/to/deploy'
            }
        }
    }
    
    post {
        success {
            echo 'Pipeline completed successfully!'
        }
        failure {
            echo 'Pipeline failed. Please check the logs.'
        }
    }
}
