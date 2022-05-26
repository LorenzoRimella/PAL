import numpy as np


def K_eta_SEIR( beta, rho, gamma):
    
    def K_eta_matrix(x_t):

        matrix      = np.zeros((4, 4))

        matrix[0,0] = np.exp(-beta*x_t[2]/(np.sum(x_t)))
        matrix[0,1] = 1 - np.exp(-beta*x_t[2]/(np.sum(x_t)))

        matrix[1,1+0] = np.exp(-rho)
        matrix[1,1+1] = 1 - np.exp(-rho)

        matrix[2,2+0] = np.exp(-gamma)
        matrix[2,2+1] = 1 - np.exp(-gamma)

        matrix[3,3]   = 1

        return matrix

    return K_eta_matrix

def K_eta_SEEIR( beta, rho_1, rho_2, gamma):
    
    def K_eta_matrix(x_t):

        matrix      = np.zeros((5, 5))

        matrix[0,0] = np.exp(-beta*x_t[3]/(np.sum(x_t)))
        matrix[0,1] = 1 - np.exp(-beta*x_t[3]/(np.sum(x_t)))

        matrix[1,1+0] = np.exp(-rho_1)
        matrix[1,1+1] = 1 - np.exp(-rho_1)

        matrix[2,2+0] = np.exp(-rho_2)
        matrix[2,2+1] = 1 - np.exp(-rho_2)

        matrix[3,3+0] = np.exp(-gamma)
        matrix[3,3+1] = 1 - np.exp(-gamma)

        matrix[4,4]   = 1

        return matrix

    return K_eta_matrix


class Compartmental_model():

    def __init__(self, pi_0, delta, K_eta, alpha, q, G, kappa, n):

        self.n     = n
        self.pi_0  = pi_0
        self.delta = delta
        self.K_eta = K_eta
        self.alpha = alpha
        self.q     = q    
        self.G     = G    
        self.kappa = kappa

    def step_0(self):

        return np.transpose(np.random.multinomial(self.n, self.pi_0.squeeze(), size = 1))
        # return (np.random.poisson(self.n*self.pi_0))

    def step_t(self, x_tm1):

        # Latent process
        # deaths
        # deaths    = np.random.binomial(x_tm1, (1-self.delta))
        barx_tm1  = np.random.binomial(x_tm1, self.delta)

        # transitions
        K_eta_tm1     = self.K_eta(barx_tm1)
        transitions_x = np.array([np.random.multinomial(barx_tm1[i], K_eta_tm1[i,:], 1)[0] for i in range(0, len(barx_tm1))])
        tildex_t      = np.transpose(np.sum(transitions_x, axis = 0, keepdims=True))

        # births
        births = np.random.poisson(self.alpha)
        x_t    = tildex_t + births

        # Observed process
        bary_t        = np.random.binomial(x_t, self.q)
        transitions_y = np.array([np.random.multinomial(bary_t[i], self.G[i,:], 1)[0] for i in range(0, len(bary_t))])
        tildey_t      = np.transpose(np.sum(transitions_y, axis = 0, keepdims=True))

        # clutter
        haty_t = np.random.poisson(self.kappa)
        y_t    = tildey_t + haty_t

        return x_t, y_t

    def run(self, T):

        X = self.step_0()
        Y = -np.ones(X.shape)

        for t in range(0, T):

            x_t, y_t = self.step_t(X[:,t:t+1])

            X = np.concatenate((X, x_t), axis = 1)
            Y = np.concatenate((Y, y_t), axis = 1)

        return X, Y

    def prediction(self, barlambda_tm1):
    
        K_eta_tm1 = self.K_eta(barlambda_tm1*self.delta)

        return np.transpose(np.dot(np.transpose(barlambda_tm1*self.delta), K_eta_tm1)) + self.alpha/self.n

    def update(self, nu_t):

        return np.transpose(np.dot(np.transpose(nu_t*self.q), self.G)) + self.kappa/self.n

    def run_nu(self, T):
        
        Nu = self.pi_0
        Nu_pred = -np.ones(self.pi_0.shape)

        for t in range(0, T):

            nu_t = self.prediction(Nu[:,t:t+1])
            nu_t_pred = self.update(nu_t)

            Nu = np.concatenate((Nu, nu_t), axis = 1)
            Nu_pred = np.concatenate((Nu_pred, nu_t_pred), axis = 1)

        return Nu, Nu_pred
        
class PoissonApproximation():

    def __init__(self, pi_0, delta, K_eta, alpha, q, G, kappa, n):
        
        self.pi_0     = pi_0
        self.lambda_0 = n*pi_0
        self.delta = delta
        self.K_eta = K_eta
        self.alpha = alpha
        self.q     = q    
        self.G     = G    
        self.kappa = kappa
        self.n     = n

    def prediction(self, barlambda_tm1):

        K_eta_tm1 = self.K_eta(barlambda_tm1*self.delta)
        lambda_t  = np.transpose(np.dot(np.transpose(barlambda_tm1*self.delta), K_eta_tm1)) + self.alpha

        return lambda_t

    def update(self, lambda_t, y_t):

        barlambda_t = (1 - self.q + np.transpose(np.dot(np.transpose(y_t)/(np.dot(np.transpose(self.q*lambda_t), self.G) + np.transpose(self.kappa)), (np.transpose(self.q)*np.transpose(self.G)))))*lambda_t

        logw_t = - np.sum(np.transpose(np.dot(np.transpose(self.q*lambda_t), self.G)) + self.kappa) + np.sum(y_t*np.log(np.transpose(np.dot(np.transpose(self.q*lambda_t), self.G)) + self.kappa)) 
        logw_t = logw_t - np.sum([np.sum(np.log(np.linspace(1, y_t[i,0], int(y_t[i,0])))) for i in range(0, y_t.shape[0])])

        return barlambda_t, np.array([[logw_t]])

    def step_t(self, barlambda_tm1, y_t):
        
        lambda_t    = self.prediction(barlambda_tm1)
        barlambda_t, logw_t = self.update(lambda_t, y_t)

        return lambda_t, barlambda_t, logw_t

    def run(self, Y):

        Lambda    = self.lambda_0
        barLambda = self.lambda_0
        logW   = -np.ones((1,1))

        for t in range(0, Y.shape[1]-1):

            lambda_t, barlambda_t, logw_t = self.step_t(barLambda[:,t:t+1], Y[:, t+1:t+2])

            Lambda = np.concatenate((Lambda, lambda_t), axis = 1)
            barLambda = np.concatenate((barLambda, barlambda_t), axis = 1)
            logW   = np.concatenate((logW,   logw_t),      axis = 1)

        return Lambda, barLambda, logW


    def update_mu(self, lambda_t):
    
        return np.transpose(np.dot(np.transpose(lambda_t*self.q), self.G)) + self.kappa

    def run_mu_lambda(self, Y):

        Lambda    = self.lambda_0
        barLambda = self.lambda_0
        Mu        = -np.ones(Lambda.shape)

        for t in range(0, Y.shape[1]-1):

            lambda_t, barlambda_t, _ = self.step_t(barLambda[:,t:t+1], Y[:, t+1:t+2])

            mu_t = self.update_mu(lambda_t)

            Lambda    = np.concatenate((Lambda, lambda_t), axis = 1)
            barLambda = np.concatenate((barLambda, barlambda_t), axis = 1)
            Mu        = np.concatenate((Mu,     mu_t),      axis = 1)

        return Lambda, barLambda, Mu

    def prediction_infty(self, barlambda_tm1):
    
        K_eta_tm1 = self.K_eta(barlambda_tm1*self.delta)
        lambda_t  = np.transpose(np.dot(np.transpose(barlambda_tm1*self.delta), K_eta_tm1)) + self.alpha/self.n

        return lambda_t

    def update_mu_infty(self, lambda_t):
        
        return np.transpose(np.dot(np.transpose(lambda_t*self.q), self.G)) + self.kappa/self.n

    def update_infty(self, mu_t_true, mu_t, lambda_t):

        barlambda_t = (1 - self.q + np.transpose(np.dot(np.transpose(mu_t_true/mu_t), (np.transpose(self.q)*np.transpose(self.G)) )))*lambda_t

        return barlambda_t

    def run_mu_lambda_infty(self, pi_0_true, delta_true, K_eta_true, alpha_true, q_true, G_true, kappa_true, T):

        def true_prediction_infty(barlambda_tm1):
        
            K_eta_tm1 = K_eta_true(barlambda_tm1*delta_true)
            lambda_t  = np.transpose(np.dot(np.transpose(barlambda_tm1*delta_true), K_eta_tm1)) + alpha_true/self.n

            return lambda_t

        def true_update_mu_infty(lambda_t):
            
            return np.transpose(np.dot(np.transpose(lambda_t*q_true), G_true)) + kappa_true/self.n

        def true_update_infty(mu_t_true, lambda_t):
        
            barlambda_t = (1 - q_true + np.transpose(np.dot(np.transpose(mu_t_true/mu_t_true), (np.transpose(q_true)*np.transpose(G_true)) )))*lambda_t

            return barlambda_t

        Lambda       = self.pi_0
        Lambda_true  = pi_0_true

        barLambda       = self.pi_0
        barLambda_true  = pi_0_true

        Mu      = -np.ones(Lambda.shape)
        Mu_true = -np.ones(Lambda.shape)

        for t in range(T):
            
            lambda_t      = self.prediction_infty(barLambda[:,t:t+1])
            lambda_t_true = true_prediction_infty(barLambda_true[:,t:t+1])

            mu_t      = self.update_mu_infty(lambda_t)
            mu_t_true = true_update_mu_infty(lambda_t_true)

            barlambda_t      = self.update_infty(mu_t_true, mu_t, lambda_t)
            barlambda_t_true = true_update_infty(mu_t_true, lambda_t_true)

            Lambda    = np.concatenate((Lambda, lambda_t), axis = 1)
            barLambda = np.concatenate((barLambda, barlambda_t), axis = 1)
            Mu        = np.concatenate((Mu,     mu_t),        axis = 1)           

            Lambda_true    = np.concatenate((Lambda_true, lambda_t_true), axis = 1)
            barLambda_true = np.concatenate((barLambda_true, barlambda_t_true), axis = 1)
            Mu_true        = np.concatenate((Mu_true,     mu_t_true),        axis = 1)

        return Lambda, Lambda_true, barLambda, barLambda_true, Mu, Mu_true


