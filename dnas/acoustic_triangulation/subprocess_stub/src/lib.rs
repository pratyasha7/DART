use std::io::{self, Write};

pub struct Exec;
pub struct Redirection;
pub struct ExitStatus;

impl ExitStatus {
    pub fn success(&self) -> bool {
        true
    }
}

pub struct DummyWriter;

impl Write for DummyWriter {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        Ok(buf.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

pub struct Popen {
    pub stdin: Option<DummyWriter>,
}

impl Exec {
    pub fn cmd<T>(_cmd: T) -> Self {
        Exec
    }
    pub fn arg<T>(&mut self, _arg: T) -> &mut Self {
        self
    }
    pub fn stdin<T>(&mut self, _to: T) -> &mut Self {
        self
    }
    pub fn stdout<T>(&mut self, _to: T) -> &mut Self {
        self
    }
    pub fn stderr<T>(&mut self, _to: T) -> &mut Self {
        self
    }
    pub fn popen(&mut self) -> Result<Popen, ()> {
        Ok(Popen {
            stdin: Some(DummyWriter),
        })
    }
    pub fn capture(&self) -> Result<(), ()> {
        Ok(())
    }
}

impl Popen {
    pub fn communicate<T>(&mut self, _input: Option<T>) -> Result<(Option<Vec<u8>>, Option<Vec<u8>>), ()> {
        Ok((None, None))
    }
    pub fn wait(&mut self) -> Result<ExitStatus, ()> {
        Ok(ExitStatus)
    }
}

impl Redirection {
    #[allow(non_snake_case)]
    pub fn Pipe() -> Self {
        Redirection
    }
    #[allow(non_snake_case)]
    pub fn Merge() -> Self {
        Redirection
    }
}