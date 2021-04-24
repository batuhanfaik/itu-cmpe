module Main where
import System.Environment ( getArgs )
import Data.List ()  

main = do
    args <- getArgs
    putStrLn "Args: "
    mapM putStrLn args