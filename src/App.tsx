import { Center } from "@chakra-ui/react";
import { Header } from "./components";
import { Projection } from "./components";
import { useData } from "./hooks";

const App = () => {
  console.log("App");
  const { loading, data } = useData("/projections.json");
  if (data === null || loading) {
    return <Center>Loading...</Center>;
  } else {
    return (
      <Center w="full" flexDir={"column"} p={0} m={0} overflowY={"hidden"}>
        <Header />

        <Projection
          data={data}
          width={500}
          height={500}
          margin={{ top: 20, right: 20, bottom: 20, left: 20 }}
        />
      </Center>
    );
  }
};

export default App;
