module Main where
import System.Environment(getArgs)
import Data.Char(intToDigit, digitToInt)

parseArgs :: [String] -> [Char]
parseArgs args
    | head args == "d2c" = digitsToChars (read $ args !! 1) (read $ args !! 2)
    | head args == "c2d" = charsToDigits (read $ args !! 1) (args !! 2)
    | head args == "n2l" = getNumberToList (read $ args !! 1) (read $ args !! 2)
    | head args == "l2n" = getListToNumber (read $ args !! 1) (map read $ (tail . tail) args)
    | head args == "add" = addDecimals (read $ args !! 1) (read $ args !! 2) (read $ args !! 3)
    | otherwise = "Unrecognized operation!"

digitsToChars :: Int -> Int -> [Char]
digitsToChars base dVal
    | base <= dVal = "Invalid digit!"
    | otherwise = show $ mapDecToHex dVal

mapDecToHex :: Int -> Char
mapDecToHex x
    | x < 10 = intToDigit x
    | x == 10 = 'A'
    | x == 11 = 'B'
    | x == 12 = 'C'
    | x == 13 = 'D'
    | x == 14 = 'E'
    | x == 15 = 'F'


charsToDigits :: Int -> [Char] -> [Char]
charsToDigits base cVal
    | base <= read (mapHexToDec cVal) = error "Invalid digit!"
    | otherwise = mapHexToDec cVal

mapHexToDec :: [Char] -> [Char]
mapHexToDec x
    | x == "A" = "10"
    | x == "B" = "11"
    | x == "C" = "12"
    | x == "D" = "13"
    | x == "E" = "14"
    | x == "F" = "15"
    | otherwise = x

getNumberToList :: Int -> Int -> [Char]
getNumberToList base num = show $ reverse $ numberToList base num

numberToList :: Int -> Int -> [Int]
numberToList base num
    | base <= 0 || num <= 0 = []
    | otherwise = mod num base:numberToList base (div num base)

getListToNumber :: Int -> [Int] -> [Char]
getListToNumber base nums = show $ listToNumber base nums

listToNumber :: Int -> [Int] -> Int
listToNumber base nums
    | base <= 0 = 0
    | otherwise = baseConverter base nums (length nums - 1)
        where
            baseConverter :: Int -> [Int] -> Int -> Int
            baseConverter _ _ (-1) = 0
            baseConverter base (x:xs) pow = x*base^pow + baseConverter base xs (pow - 1)

addDecimals :: Int -> Int -> Int -> [Char]
addDecimals base num1 num2 =
    getNumberToList base num1 ++ "\n\"" ++ reverse (map mapDecToHex (numberToList base num1)) ++ "\"\n" ++
    getNumberToList base num2 ++ "\n\"" ++ reverse (map mapDecToHex (numberToList base num2)) ++ "\"\n" ++
    show (num1 + num2)

main :: IO ()
main = do
    args <- getArgs
    putStrLn $ parseArgs args